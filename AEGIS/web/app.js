"use strict";

const API = Object.freeze({
  evaluate: "/api/evaluate",
  status: "/api/status",
  summary: "/api/history/summary",
  history: "/api/history",
  settings: "/api/settings",
});

const ROUTES = Object.freeze({
  "/": {
    title: "Evaluate · AEGIS",
    heading: "evaluate-page-title",
  },
  "/history": {
    title: "History · AEGIS",
    heading: "history-page-title",
  },
  "/settings": {
    title: "Settings · AEGIS",
    heading: "settings-page-title",
  },
});
const ALLOWED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const INPUT_MODALITIES = new Set(["text", "image", "image_text"]);
const DEFAULT_TEXT_LIMIT = 32_768;
const DEFAULT_IMAGE_LIMIT = 8 * 1024 * 1024;
const SETTINGS_POLL_INTERVAL_MS = 1500;
const GUI_SESSION_HEADER = "X-AEGIS-GUI-Session";
const GUI_SESSION_STORAGE_KEY = "aegis.gui.session.v1";
const GUI_SESSION_PATTERN = /^[A-Za-z0-9_-]{32,}$/;
const guiSessionToken = establishGuiSession();

const state = {
  activeRoute: "/",
  selectedFile: null,
  previewUrl: null,
  imageLimit: DEFAULT_IMAGE_LIMIT,
  textLimit: DEFAULT_TEXT_LIMIT,
  serviceReady: false,
  supportedModalities: null,
  targetProfile: null,
  evaluating: false,
  historyOffset: 0,
  historyLimit: 25,
  historyTotal: 0,
  historyRequest: 0,
  historyLoaded: false,
  historyDirty: true,
  filterTimer: null,
  clearingHistory: false,
  settings: null,
  settingsLoaded: false,
  settingsRequest: 0,
  settingsSubmitting: false,
  settingsDesired: null,
  settingsPollTimer: null,
  dialogReturnFocus: null,
  historyImageUrls: [],
  historyImageRequest: 0,
};

const elements = {
  navLinks: Array.from(document.querySelectorAll(".nav-link[data-route]")),
  pageViews: Array.from(document.querySelectorAll(".page-view[data-page]")),
  evaluateForm: document.querySelector("#evaluate-form"),
  promptText: document.querySelector("#prompt-text"),
  characterCount: document.querySelector("#character-count"),
  promptHint: document.querySelector("#prompt-hint"),
  promptError: document.querySelector("#prompt-error"),
  imageInput: document.querySelector("#image-input"),
  imageError: document.querySelector("#image-error"),
  imageRequirement: document.querySelector("#image-requirement"),
  dropZone: document.querySelector("#drop-zone"),
  dropHelp: document.querySelector("#drop-help"),
  filePreview: document.querySelector("#file-preview"),
  fileThumbnail: document.querySelector("#file-thumbnail"),
  fileName: document.querySelector("#file-name"),
  fileMeta: document.querySelector("#file-meta"),
  removeImage: document.querySelector("#remove-image"),
  evaluateButton: document.querySelector("#evaluate-button"),
  buttonLabel: document.querySelector("#evaluate-button .button-label"),
  modalityGuidance: document.querySelector("#modality-guidance"),
  resultState: document.querySelector("#result-state"),
  resultContainer: document.querySelector("#result-container"),
  refreshStatus: document.querySelector("#refresh-status"),
  sidebarServiceState: document.querySelector("#sidebar-service-state"),
  sidebarStatusText: document.querySelector("#sidebar-status-text"),
  sidebarModel: document.querySelector("#sidebar-model"),
  headerServiceState: document.querySelector("#header-service-state"),
  headerStatusText: document.querySelector("#header-status-text"),
  kpiTotal: document.querySelector("#kpi-total"),
  kpiAllow: document.querySelector("#kpi-allow"),
  kpiReview: document.querySelector("#kpi-review"),
  kpiBlock: document.querySelector("#kpi-block"),
  historyFilters: document.querySelector("#history-filters"),
  historySearch: document.querySelector("#history-search"),
  filterAction: document.querySelector("#filter-action"),
  filterVerdict: document.querySelector("#filter-verdict"),
  filterModality: document.querySelector("#filter-modality"),
  filterStatus: document.querySelector("#filter-status"),
  filterFrom: document.querySelector("#filter-from"),
  filterTo: document.querySelector("#filter-to"),
  filterLimit: document.querySelector("#filter-limit"),
  clearFilters: document.querySelector("#clear-filters"),
  refreshHistory: document.querySelector("#refresh-history"),
  historyBody: document.querySelector("#history-body"),
  historyMessage: document.querySelector("#history-message"),
  historyCount: document.querySelector("#history-count"),
  activeFilterNote: document.querySelector("#active-filter-note"),
  previousPage: document.querySelector("#previous-page"),
  nextPage: document.querySelector("#next-page"),
  pageStatus: document.querySelector("#page-status"),
  historyDialog: document.querySelector("#history-dialog"),
  closeDialog: document.querySelector("#close-dialog"),
  dialogContent: document.querySelector("#dialog-content"),
  clearHistory: document.querySelector("#clear-history"),
  clearHistoryDialog: document.querySelector("#clear-history-dialog"),
  clearHistoryError: document.querySelector("#clear-history-error"),
  cancelClearHistory: document.querySelector("#cancel-clear-history"),
  confirmClearHistory: document.querySelector("#confirm-clear-history"),
  settingsForm: document.querySelector("#settings-form"),
  settingsState: document.querySelector("#settings-state"),
  settingsControlNote: document.querySelector("#settings-control-note"),
  settingsError: document.querySelector("#settings-error"),
  targetOptions: document.querySelector("#target-options"),
  selectedTargetTitle: document.querySelector("#selected-target-title"),
  selectedTargetStatus: document.querySelector("#selected-target-status"),
  selectedTargetDescription: document.querySelector("#selected-target-description"),
  selectedTargetModel: document.querySelector("#selected-target-model"),
  selectedTargetInputs: document.querySelector("#selected-target-inputs"),
  selectedTargetCaveats: document.querySelector("#selected-target-caveats"),
  selectedTargetCaveatList: document.querySelector("#selected-target-caveat-list"),
  shadowMode: document.querySelector("#shadow-mode"),
  shadowModeHelp: document.querySelector("#shadow-mode-help"),
  enforceWarning: document.querySelector("#enforce-warning"),
  applySettings: document.querySelector("#apply-settings"),
  applySettingsLabel: document.querySelector("#apply-settings .button-label"),
  currentTargetProfile: document.querySelector("#current-target-profile"),
  currentModelId: document.querySelector("#current-model-id"),
  currentTrafficMode: document.querySelector("#current-traffic-mode"),
  currentReadiness: document.querySelector("#current-readiness"),
  enforceDialog: document.querySelector("#enforce-dialog"),
  enforceDialogDescription: document.querySelector("#enforce-dialog-description"),
  enforceDialogError: document.querySelector("#enforce-dialog-error"),
  cancelEnforce: document.querySelector("#cancel-enforce"),
  confirmEnforce: document.querySelector("#confirm-enforce"),
  toastRegion: document.querySelector("#toast-region"),
};

document.addEventListener("DOMContentLoaded", initialize);

function initialize() {
  bindEvents();
  updateCharacterCount();
  updateEvaluateAvailability();
  setHistoryLoading();
  setSettingsLoading();
  activateRoute(routeForPath(window.location.pathname), { initial: true });
  loadStatus();
}

function establishGuiSession() {
  let token = "";
  const fragment = window.location.hash.startsWith("#")
    ? window.location.hash.slice(1)
    : "";
  if (fragment) {
    const params = new URLSearchParams(fragment);
    const candidates = params.getAll("session");
    const keys = Array.from(params.keys());
    if (
      candidates.length === 1 &&
      keys.length === 1 &&
      keys[0] === "session" &&
      GUI_SESSION_PATTERN.test(candidates[0])
    ) {
      token = candidates[0];
      try {
        window.sessionStorage.setItem(GUI_SESSION_STORAGE_KEY, token);
      } catch {
        // The in-memory token remains usable when browser storage is unavailable.
      }
    }
    if (params.has("session")) {
      window.history.replaceState(
        {},
        "",
        `${window.location.pathname}${window.location.search}`,
      );
    }
  }
  if (!token) {
    try {
      const stored = window.sessionStorage.getItem(GUI_SESSION_STORAGE_KEY) ?? "";
      if (GUI_SESSION_PATTERN.test(stored)) {
        token = stored;
      } else if (stored) {
        window.sessionStorage.removeItem(GUI_SESSION_STORAGE_KEY);
      }
    } catch {
      // A fresh launch fragment is required when browser storage is unavailable.
    }
  }
  return token;
}

function bindEvents() {
  elements.navLinks.forEach((link) => {
    link.addEventListener("click", handleNavigationClick);
  });
  window.addEventListener("popstate", () => {
    activateRoute(routeForPath(window.location.pathname), { focusHeading: true });
  });
  elements.promptText.addEventListener("input", handlePromptInput);
  elements.evaluateForm.addEventListener("submit", submitEvaluation);
  elements.imageInput.addEventListener("change", handleFileInput);
  elements.removeImage.addEventListener("click", clearSelectedFile);
  elements.dropZone.addEventListener("keydown", handleDropZoneKeydown);
  elements.dropZone.addEventListener("dragenter", handleDragEnter);
  elements.dropZone.addEventListener("dragover", handleDragOver);
  elements.dropZone.addEventListener("dragleave", handleDragLeave);
  elements.dropZone.addEventListener("drop", handleDrop);
  elements.refreshStatus.addEventListener("click", () => {
    loadStatus();
    if (state.activeRoute === "/settings") {
      loadSettings();
    }
  });
  elements.refreshHistory.addEventListener("click", () => {
    loadHistory();
    loadSummary();
  });
  elements.historyFilters.addEventListener("submit", applyFilters);
  elements.clearFilters.addEventListener("click", resetFilters);
  elements.previousPage.addEventListener("click", previousPage);
  elements.nextPage.addEventListener("click", nextPage);
  elements.closeDialog.addEventListener("click", closeHistoryDialog);
  elements.historyDialog.addEventListener("click", handleDialogBackdropClick);
  elements.clearHistory.addEventListener("click", openClearHistoryDialog);
  elements.cancelClearHistory.addEventListener("click", closeClearHistoryDialog);
  elements.confirmClearHistory.addEventListener("click", confirmHistoryClear);
  elements.clearHistoryDialog.addEventListener("click", handleConfirmBackdropClick);
  elements.clearHistoryDialog.addEventListener("close", restoreDialogFocus);
  elements.clearHistoryDialog.addEventListener("cancel", (event) => {
    if (state.clearingHistory) {
      event.preventDefault();
    }
  });
  elements.settingsForm.addEventListener("submit", submitSettings);
  elements.targetOptions.addEventListener("change", handleTargetSelection);
  elements.shadowMode.addEventListener("change", handleShadowModeChange);
  elements.cancelEnforce.addEventListener("click", closeEnforceDialog);
  elements.confirmEnforce.addEventListener("click", confirmEnforcement);
  elements.enforceDialog.addEventListener("click", handleConfirmBackdropClick);
  elements.enforceDialog.addEventListener("close", restoreDialogFocus);
  elements.enforceDialog.addEventListener("cancel", (event) => {
    if (state.settingsSubmitting) {
      event.preventDefault();
    }
  });
  window.addEventListener("beforeunload", () => {
    releasePreviewUrl();
    clearTimeout(state.settingsPollTimer);
  });

  elements.historySearch.addEventListener("input", scheduleFilterUpdate);
  [
    elements.filterAction,
    elements.filterVerdict,
    elements.filterModality,
    elements.filterStatus,
    elements.filterFrom,
    elements.filterTo,
    elements.filterLimit,
  ].forEach((control) => control.addEventListener("change", applyFilters));
}

function routeForPath(pathname) {
  const normalized =
    pathname.length > 1 && pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
  return Object.hasOwn(ROUTES, normalized) ? normalized : "/";
}

function handleNavigationClick(event) {
  if (
    event.defaultPrevented ||
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) {
    return;
  }
  const link = event.currentTarget;
  const route = routeForPath(new URL(link.href, window.location.href).pathname);
  event.preventDefault();
  if (window.location.pathname !== route) {
    window.history.pushState({}, "", route);
  }
  activateRoute(route, { focusHeading: true });
}

function activateRoute(route, options = {}) {
  const nextRoute = Object.hasOwn(ROUTES, route) ? route : "/";
  if (route !== nextRoute || window.location.pathname !== nextRoute) {
    window.history.replaceState({}, "", nextRoute);
  }
  state.activeRoute = nextRoute;
  elements.pageViews.forEach((view) => {
    view.hidden = view.dataset.page !== nextRoute;
  });
  elements.navLinks.forEach((link) => {
    const active = link.dataset.route === nextRoute;
    link.classList.toggle("is-active", active);
    if (active) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  document.title = ROUTES[nextRoute].title;

  if (nextRoute !== "/history" && elements.historyDialog.open) {
    closeHistoryDialog();
  }
  if (nextRoute === "/history" && (!state.historyLoaded || state.historyDirty)) {
    Promise.allSettled([loadSummary(), loadHistory()]);
  }
  if (nextRoute === "/settings") {
    loadSettings();
  }

  if (!options.initial) {
    window.scrollTo({ top: 0, behavior: "auto" });
  }
  if (options.focusHeading) {
    document.querySelector(`#${ROUTES[nextRoute].heading}`)?.focus({
      preventScroll: true,
    });
  }
}

async function loadStatus() {
  state.serviceReady = false;
  updateEvaluateAvailability();
  setServiceStatus("pending", "Checking");
  try {
    const payload = await requestJson(API.status);
    const service = payload.service ?? payload;
    const readiness = service.readiness ?? payload.readiness ?? {};
    const ready =
      service.ready ?? payload.ready ?? readiness.ok ?? payload.ok ?? payload.status === "ready";
    const model =
      readiness.model_id ??
      service.model_id ??
      payload.model_id ??
      payload.model ??
      payload.model_name ??
      payload.provider?.model_id ??
      payload.details?.model_id;
    const mode =
      readiness.traffic_mode ??
      service.traffic_mode ??
      payload.traffic_mode ??
      payload.mode ??
      payload.details?.traffic_mode;

    state.serviceReady = Boolean(ready);
    applyServerCapabilities(payload);
    if (ready) {
      setServiceStatus("online", mode ? `Online · ${humanize(mode)}` : "Online");
    } else {
      const statusMessage =
        service.error?.message ??
        payload.error?.message ??
        (service.connected === false ? "Unavailable" : "Not ready");
      setServiceStatus("offline", statusMessage);
    }
    elements.sidebarModel.textContent =
      service.connected === false
        ? "Unavailable"
        : model
          ? `${String(model)}${mode ? ` · ${humanize(mode)}` : ""}`
          : "AEGIS";

    applyServerLimits(payload);
  } catch (error) {
    state.serviceReady = false;
    state.supportedModalities = null;
    state.targetProfile = null;
    updateCapabilityGuidance();
    setServiceStatus("offline", "Unavailable");
    elements.sidebarModel.textContent = readableError(error);
  } finally {
    updateEvaluateAvailability();
  }
}

function setServiceStatus(tone, message) {
  [elements.sidebarServiceState, elements.headerServiceState].forEach((container) => {
    const dot = container.querySelector(".status-dot");
    dot.className = `status-dot is-${tone}`;
  });
  elements.sidebarStatusText.textContent = message;
  elements.headerStatusText.textContent =
    tone === "online" ? message : tone === "pending" ? "Connecting" : "Offline";
}

function applyServerLimits(payload) {
  const limits = payload.limits ?? payload.request_limits ?? payload;
  const textLimit = toFiniteNumber(
    limits.max_text_characters ?? limits.max_text_length ?? limits.text_limit,
  );
  const imageLimit = toFiniteNumber(
    limits.effective_max_image_bytes ?? limits.max_image_bytes ?? limits.image_limit,
  );
  const bodyLimit = toFiniteNumber(limits.max_body_bytes ?? payload.max_body_bytes);

  if (textLimit && textLimit > 0) {
    state.textLimit = Math.floor(textLimit);
    elements.promptText.maxLength = state.textLimit;
  }

  if (imageLimit && imageLimit > 0) {
    state.imageLimit = Math.floor(imageLimit);
  }
  if (bodyLimit && bodyLimit > 0) {
    const base64SafeLimit = Math.max(1, Math.floor((bodyLimit - 4096) * 0.75));
    state.imageLimit = Math.min(state.imageLimit, base64SafeLimit);
  }

  elements.dropHelp.textContent =
    `Browse · PNG/JPEG/WebP · ${formatBytes(state.imageLimit)} max`;
  updateCharacterCount();
}

function applyServerCapabilities(payload) {
  const service = payload.service ?? payload;
  const readiness = service.readiness ?? payload.readiness ?? {};
  const capabilities =
    service.capabilities ?? readiness.capabilities ?? payload.capabilities ?? {};
  const rawModalities =
    capabilities.input_modalities ?? readiness.supported_modalities;

  if (
    Array.isArray(rawModalities) &&
    rawModalities.length > 0 &&
    rawModalities.every((item) => INPUT_MODALITIES.has(String(item)))
  ) {
    state.supportedModalities = new Set(rawModalities.map(String));
  } else {
    state.supportedModalities = null;
  }
  state.targetProfile =
    typeof readiness.target_profile === "string" && readiness.target_profile
      ? readiness.target_profile
      : null;
  updateCapabilityGuidance();
}

function updateCapabilityGuidance() {
  if (!state.serviceReady) {
    elements.modalityGuidance.textContent = "Service offline.";
    elements.modalityGuidance.dataset.tone = "offline";
    elements.promptHint.textContent = "Await service.";
    elements.imageRequirement.textContent = "Offline";
    elements.promptText.setAttribute("aria-required", "false");
    elements.imageInput.setAttribute("aria-required", "false");
    return;
  }

  delete elements.modalityGuidance.dataset.tone;
  if (state.supportedModalities === null) {
    elements.modalityGuidance.textContent = "Inputs validated on submit.";
    elements.promptHint.textContent = "Text, image, or both.";
    elements.imageRequirement.textContent = "Optional";
    elements.promptText.setAttribute("aria-required", "false");
    elements.imageInput.setAttribute("aria-required", "false");
    return;
  }

  const supported = Array.from(state.supportedModalities);
  const target = state.targetProfile ? `${state.targetProfile}: ` : "";
  const imageOnlyNote = state.supportedModalities.has("image")
    ? ""
    : " · No image-only";
  elements.modalityGuidance.textContent =
    `${target}${formatInputModalities(supported)}${imageOnlyNote}`;

  const promptRequired = supported.every((modality) => modality !== "image");
  const imageRequired = supported.every(
    (modality) => modality === "image" || modality === "image_text",
  );
  elements.promptText.setAttribute("aria-required", String(promptRequired));
  elements.imageInput.setAttribute("aria-required", String(imageRequired));
  elements.imageRequirement.textContent = imageRequired ? "Required" : "Optional";
  elements.promptHint.textContent = promptRequired
    ? "Text required."
    : "Text optional.";
}

function formatInputModalities(modalities) {
  const labels = modalities.map(
    (modality) =>
      ({
        text: "Text",
        image: "Image",
        image_text: "Text + image",
      })[modality] ?? modality,
  );
  if (labels.length === 1) {
    return labels[0];
  }
  if (labels.length === 2) {
    return `${labels[0]} or ${labels[1]}`;
  }
  return `${labels.slice(0, -1).join(", ")}, or ${labels.at(-1)}`;
}

async function loadSettings(options = {}) {
  const requestId = ++state.settingsRequest;
  if (!state.settingsLoaded && !options.polling) {
    setSettingsLoading();
  }
  try {
    const payload = await requestJson(API.settings);
    if (requestId !== state.settingsRequest) {
      return;
    }
    const wasBusy = Boolean(state.settings?.targetControl.busy);
    renderSettings(payload);
    state.settingsLoaded = true;
    if (wasBusy && !state.settings.targetControl.busy) {
      state.settingsDesired = null;
      renderSettings(payload);
      await loadStatus();
      showToast(
        "Applied",
        "Target and mode ready.",
        "success",
      );
    }
  } catch (error) {
    if (requestId !== state.settingsRequest) {
      return;
    }
    renderSettingsError(readableError(error));
  }
}

function normalizeSettings(payload) {
  const currentPayload =
    payload?.current && typeof payload.current === "object" ? payload.current : {};
  const controlPayload =
    payload?.target_control && typeof payload.target_control === "object"
      ? payload.target_control
      : {};
  const targets = Array.isArray(payload?.targets)
    ? payload.targets
        .filter((target) => target && typeof target === "object" && target.name)
        .map((target) => ({
          name: String(target.name),
          title: String(target.title ?? target.name),
          description: String(target.description ?? "No description."),
          modelId: String(target.model_id ?? "—"),
          inputModalities: Array.isArray(target.input_modalities)
            ? target.input_modalities
                .map(String)
                .filter((modality) => INPUT_MODALITIES.has(modality))
            : [],
          trafficMode: String(target.traffic_mode ?? "shadow"),
          status: String(target.status ?? "unknown"),
          detectorMode: String(target.detector_mode ?? "single"),
          caveats: toStringArray(target.caveats),
        }))
    : [];
  const trafficModes = Array.isArray(payload?.traffic_modes)
    ? payload.traffic_modes.map(String)
    : ["shadow", "enforce"];
  return {
    current: {
      targetProfile:
        currentPayload.target_profile === null ||
        currentPayload.target_profile === undefined
          ? ""
          : String(currentPayload.target_profile),
      trafficMode: String(currentPayload.traffic_mode ?? "shadow"),
      modelId: String(currentPayload.model_id ?? "—"),
      ready: Boolean(currentPayload.ready),
      detectorMode: String(currentPayload.detector_mode ?? "single"),
      runtimeTrafficModeControl: Boolean(
        currentPayload.runtime_traffic_mode_control,
      ),
    },
    targets,
    trafficModes,
    targetControl: {
      available: Boolean(controlPayload.available),
      busy: Boolean(controlPayload.busy),
      reason:
        controlPayload.reason === null || controlPayload.reason === undefined
          ? ""
          : String(controlPayload.reason),
    },
  };
}

function setSettingsLoading() {
  elements.settingsForm.setAttribute("aria-busy", "true");
  elements.settingsState.textContent = "Loading";
  elements.settingsState.dataset.tone = "pending";
  elements.settingsControlNote.textContent = "Loading controls…";
  elements.settingsError.textContent = "";
  elements.applySettings.disabled = true;
  elements.shadowMode.disabled = true;
}

function renderSettings(payload) {
  const settings = normalizeSettings(payload);
  state.settings = settings;
  const desired = settings.targetControl.busy ? state.settingsDesired : null;
  if (!settings.targetControl.busy) {
    state.settingsDesired = null;
  }
  const selectedTarget =
    desired?.targetProfile || settings.current.targetProfile || settings.targets[0]?.name || "";
  const selectedMode =
    desired?.trafficMode || trafficModeForTarget(settings, selectedTarget);

  clearNode(elements.targetOptions);
  if (!settings.targets.length) {
    const empty = createElement("div", "settings-empty");
    empty.append(createElement("strong", "", "No targets"));
    empty.append(
      createElement(
        "p",
        "",
        "Prepare a target to continue.",
      ),
    );
    elements.targetOptions.append(empty);
  } else {
    settings.targets.forEach((target, index) => {
      elements.targetOptions.append(
        createTargetOption(target, index, selectedTarget, settings.current.targetProfile),
      );
    });
  }

  elements.shadowMode.checked = selectedMode === "shadow";
  elements.currentTargetProfile.textContent =
    settings.current.targetProfile || "—";
  elements.currentModelId.textContent = settings.current.modelId;
  elements.currentTrafficMode.textContent = humanize(settings.current.trafficMode);
  elements.currentReadiness.textContent = settings.current.ready ? "Ready" : "Not ready";
  elements.currentReadiness.dataset.tone = settings.current.ready ? "allow" : "review";
  elements.settingsError.textContent = "";
  renderSelectedTargetDetails();
  updateSettingsInterface();

  if (settings.targetControl.busy) {
    scheduleSettingsPoll();
  } else {
    clearTimeout(state.settingsPollTimer);
    state.settingsPollTimer = null;
  }
}

function createTargetOption(target, index, selectedTarget, currentTarget) {
  const label = createElement("label", "target-option");
  const input = document.createElement("input");
  input.type = "radio";
  input.name = "target_profile";
  input.value = target.name;
  input.checked = target.name === selectedTarget;
  input.setAttribute("aria-describedby", `target-option-description-${index}`);

  const body = createElement("span", "target-option-body");
  const heading = createElement("span", "target-option-heading");
  heading.append(createElement("strong", "", target.title));
  const badge = createElement(
    "span",
    "badge",
    target.name === currentTarget ? "Active" : humanize(target.status),
  );
  badge.dataset.tone = target.name === currentTarget ? "allow" : toneFor(target.status);
  heading.append(badge);
  body.append(heading);

  const description = createElement(
    "small",
    "target-option-description",
    target.description,
  );
  description.id = `target-option-description-${index}`;
  body.append(description);

  const metadata = createElement("span", "target-option-meta");
  metadata.append(
    createElement("span", "", target.modelId),
    createElement(
      "span",
      "",
      target.inputModalities.length
        ? formatInputModalities(target.inputModalities)
        : "Inputs unknown",
    ),
    createElement(
      "span",
      "",
      target.detectorMode === "or" ? "Dual-head OR" : "Single detector",
    ),
    createElement("span", "", humanize(target.trafficMode)),
  );
  body.append(metadata);
  label.append(input, body);
  return label;
}

function selectedTargetName() {
  return (
    elements.targetOptions.querySelector('input[name="target_profile"]:checked')?.value ??
    ""
  );
}

function selectedTargetRecord() {
  const name = selectedTargetName();
  return state.settings?.targets.find((target) => target.name === name) ?? null;
}

function trafficModeForTarget(settings, targetName) {
  if (targetName === settings.current.targetProfile) {
    return settings.current.trafficMode;
  }
  return (
    settings.targets.find((target) => target.name === targetName)?.trafficMode ??
    "shadow"
  );
}

function renderSelectedTargetDetails() {
  const target = selectedTargetRecord();
  clearNode(elements.selectedTargetCaveatList);
  if (!target) {
    elements.selectedTargetTitle.textContent = "Select target";
    elements.selectedTargetStatus.textContent = "Unavailable";
    elements.selectedTargetStatus.dataset.tone = "error";
    elements.selectedTargetDescription.textContent = "Choose a target.";
    elements.selectedTargetModel.textContent = "—";
    elements.selectedTargetInputs.textContent = "—";
    elements.selectedTargetCaveats.hidden = true;
    return;
  }

  elements.selectedTargetTitle.textContent = target.title;
  elements.selectedTargetStatus.textContent = humanize(target.status);
  elements.selectedTargetStatus.dataset.tone = toneFor(target.status);
  elements.selectedTargetDescription.textContent = target.description;
  elements.selectedTargetModel.textContent = target.modelId;
  elements.selectedTargetInputs.textContent = target.inputModalities.length
    ? formatInputModalities(target.inputModalities)
    : "—";
  elements.selectedTargetCaveats.hidden = target.caveats.length === 0;
  target.caveats.forEach((caveat) => {
    elements.selectedTargetCaveatList.append(createElement("li", "", caveat));
  });
}

function handleTargetSelection() {
  elements.settingsError.textContent = "";
  const targetName = selectedTargetName();
  elements.shadowMode.checked =
    trafficModeForTarget(state.settings, targetName) === "shadow";
  renderSelectedTargetDetails();
  updateSettingsInterface();
}

function handleShadowModeChange() {
  elements.settingsError.textContent = "";
  updateSettingsInterface();
}

function updateSettingsInterface() {
  const settings = state.settings;
  if (!settings) {
    elements.applySettings.disabled = true;
    elements.shadowMode.disabled = true;
    return;
  }
  const busy = settings.targetControl.busy || state.settingsSubmitting;
  const targetName = selectedTargetName();
  const swappingTarget = targetName !== settings.current.targetProfile;
  const controlAvailable =
    settings.current.runtimeTrafficModeControl &&
    (!swappingTarget || settings.targetControl.available);
  const anyControlAvailable = settings.current.runtimeTrafficModeControl;
  const selectedMode = elements.shadowMode.checked ? "shadow" : "enforce";
  const modeAvailable = settings.trafficModes.includes(selectedMode);
  const hasChanges =
    targetName !== settings.current.targetProfile ||
    selectedMode !== settings.current.trafficMode;

  elements.targetOptions
    .querySelectorAll('input[name="target_profile"]')
    .forEach((input) => {
      const isCurrent = input.value === settings.current.targetProfile;
      input.disabled =
        busy ||
        (!isCurrent &&
          (!settings.targetControl.available ||
            !settings.current.runtimeTrafficModeControl));
    });
  elements.shadowMode.disabled = busy || !controlAvailable;
  elements.applySettings.disabled =
    !controlAvailable || busy || !modeAvailable || !targetName || !hasChanges;
  elements.settingsForm.setAttribute("aria-busy", String(busy));
  elements.settingsState.textContent = busy
    ? "Switching"
    : !anyControlAvailable
      ? "Unavailable"
      : settings.targetControl.available
        ? "Available"
        : "Mode control";
  elements.settingsState.dataset.tone = busy
    ? "pending"
    : !anyControlAvailable
      ? "error"
      : "allow";
  elements.enforceWarning.hidden = elements.shadowMode.checked;
  elements.shadowModeHelp.textContent = elements.shadowMode.checked
    ? "Logs recommendations; always allows."
    : "Recommendations become actions.";

  if (busy) {
    elements.settingsControlNote.textContent = "Loading target; evaluations may pause.";
  } else if (!settings.current.runtimeTrafficModeControl) {
    elements.settingsControlNote.textContent = "Runtime mode control unavailable.";
  } else if (swappingTarget && !settings.targetControl.available) {
    elements.settingsControlNote.textContent =
      settings.targetControl.reason ||
      "Target switching unavailable.";
  } else if (!modeAvailable) {
    elements.settingsControlNote.textContent =
      `${humanize(selectedMode)} unavailable.`;
  } else if (hasChanges) {
    elements.settingsControlNote.textContent = "Ready to apply.";
  } else {
    elements.settingsControlNote.textContent = "No changes.";
  }
}

function renderSettingsError(message) {
  state.settingsLoaded = false;
  state.settingsDesired = null;
  clearTimeout(state.settingsPollTimer);
  state.settingsPollTimer = null;
  elements.settingsForm.setAttribute("aria-busy", "false");
  elements.settingsState.textContent = "Unavailable";
  elements.settingsState.dataset.tone = "error";
  elements.settingsControlNote.textContent = "Settings unavailable. Refresh to retry.";
  elements.settingsError.textContent = message;
  elements.applySettings.disabled = true;
  elements.shadowMode.disabled = true;
}

function submitSettings(event) {
  event.preventDefault();
  if (elements.applySettings.disabled || state.settingsSubmitting) {
    return;
  }
  elements.settingsError.textContent = "";
  if (!elements.shadowMode.checked) {
    openEnforceDialog();
    return;
  }
  applySettingsChange();
}

function openEnforceDialog() {
  const target = selectedTargetRecord();
  const title = target?.title ?? selectedTargetName() ?? "selected target";
  elements.enforceDialogDescription.textContent =
    `${title} will enter Enforce mode. New evaluations may be blocked or sent to review.`;
  elements.enforceDialogError.textContent = "";
  state.dialogReturnFocus = elements.applySettings;
  showConfirmationDialog(elements.enforceDialog);
  elements.cancelEnforce.focus();
}

function closeEnforceDialog() {
  if (state.settingsSubmitting) {
    return;
  }
  closeConfirmationDialog(elements.enforceDialog);
}

function confirmEnforcement() {
  if (state.settingsSubmitting) {
    return;
  }
  closeConfirmationDialog(elements.enforceDialog);
  applySettingsChange();
}

async function applySettingsChange() {
  const targetProfile = selectedTargetName();
  const trafficMode = elements.shadowMode.checked ? "shadow" : "enforce";
  if (!targetProfile) {
    elements.settingsError.textContent = "Select a target.";
    return;
  }

  state.settingsDesired = { targetProfile, trafficMode };
  setSettingsSubmitting(true);
  try {
    const payload = await requestJson(API.settings, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_profile: targetProfile,
        traffic_mode: trafficMode,
      }),
    });
    renderSettings(payload);
    state.settingsLoaded = true;
    await loadStatus();
    if (state.settings.targetControl.busy) {
      showToast(
        "Switching",
        "Loading model…",
        "info",
      );
    } else {
      state.settingsDesired = null;
      renderSettings(payload);
      showToast(
        "Applied",
        "Target and mode active.",
        "success",
      );
    }
  } catch (error) {
    state.settingsDesired = null;
    elements.settingsError.textContent = readableError(error);
    showToast("Not applied", readableError(error), "error");
  } finally {
    setSettingsSubmitting(false);
  }
}

function setSettingsSubmitting(submitting) {
  state.settingsSubmitting = submitting;
  elements.applySettings.classList.toggle("is-loading", submitting);
  elements.applySettingsLabel.textContent = submitting ? "Applying…" : "Apply";
  elements.applySettings.setAttribute("aria-busy", String(submitting));
  elements.confirmEnforce.disabled = submitting;
  elements.cancelEnforce.disabled = submitting;
  updateSettingsInterface();
}

function scheduleSettingsPoll() {
  clearTimeout(state.settingsPollTimer);
  state.settingsPollTimer = window.setTimeout(() => {
    loadSettings({ polling: true });
  }, SETTINGS_POLL_INTERVAL_MS);
}

function handlePromptInput() {
  updateCharacterCount();
  elements.promptError.textContent = "";
  updateEvaluateAvailability();
}

async function loadSummary() {
  try {
    const payload = await requestJson(API.summary);
    const counts =
      payload.recommended_action_counts ??
      payload.action_counts ??
      payload.actions ??
      {};
    const total = firstNumber(
      payload.total,
      payload.total_count,
      payload.evaluations_total,
      sumCounts(counts),
    );
    const allow = firstNumber(payload.allow, payload.allowed, counts.allow, 0);
    const review = firstNumber(payload.review, payload.reviewed, counts.review, 0);
    const block = firstNumber(payload.block, payload.blocked, counts.block, 0);

    setKpi(elements.kpiTotal, total);
    setKpi(elements.kpiAllow, allow);
    setKpi(elements.kpiReview, review);
    setKpi(elements.kpiBlock, block);
  } catch (error) {
    [
      elements.kpiTotal,
      elements.kpiAllow,
      elements.kpiReview,
      elements.kpiBlock,
    ].forEach((element) => {
      element.textContent = "—";
      element.title = readableError(error);
    });
  }
}

function setKpi(element, value) {
  const number = toFiniteNumber(value);
  element.textContent = number === null ? "—" : formatInteger(number);
  element.removeAttribute("title");
}

function updateCharacterCount() {
  const count = elements.promptText.value.length;
  elements.characterCount.textContent =
    `${formatInteger(count)} / ${formatInteger(state.textLimit)}`;
  elements.characterCount.dataset.nearLimit =
    count > state.textLimit * 0.9 ? "true" : "false";
}

function handleFileInput(event) {
  const file = event.target.files?.[0];
  if (file) {
    selectFile(file);
  }
}

function handleDropZoneKeydown(event) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.imageInput.click();
  }
}

function handleDragEnter(event) {
  event.preventDefault();
  elements.dropZone.classList.add("is-dragging");
}

function handleDragOver(event) {
  event.preventDefault();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = "copy";
  }
  elements.dropZone.classList.add("is-dragging");
}

function handleDragLeave(event) {
  if (!elements.dropZone.contains(event.relatedTarget)) {
    elements.dropZone.classList.remove("is-dragging");
  }
}

function handleDrop(event) {
  event.preventDefault();
  elements.dropZone.classList.remove("is-dragging");
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    selectFile(file);
  }
}

function selectFile(file) {
  elements.imageError.textContent = "";
  if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
    elements.imageError.textContent = "Use PNG, JPEG, or WebP.";
    elements.imageInput.value = "";
    return;
  }
  if (file.size > state.imageLimit) {
    elements.imageError.textContent =
      `${formatBytes(file.size)} exceeds the ${formatBytes(state.imageLimit)} limit.`;
    elements.imageInput.value = "";
    return;
  }

  releasePreviewUrl();
  state.selectedFile = file;
  state.previewUrl = URL.createObjectURL(file);
  elements.fileThumbnail.src = state.previewUrl;
  elements.fileThumbnail.alt = `Preview of ${file.name}`;
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${humanizeMediaType(file.type)} · ${formatBytes(file.size)}`;
  elements.filePreview.hidden = false;
  updateEvaluateAvailability();
}

function clearSelectedFile() {
  releasePreviewUrl();
  state.selectedFile = null;
  elements.imageInput.value = "";
  elements.fileThumbnail.removeAttribute("src");
  elements.fileThumbnail.alt = "";
  elements.fileName.textContent = "";
  elements.fileMeta.textContent = "";
  elements.filePreview.hidden = true;
  elements.imageError.textContent = "";
  updateEvaluateAvailability();
  elements.imageInput.focus();
}

function releasePreviewUrl() {
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
}

function currentInputModality() {
  const hasText = Boolean(elements.promptText.value.trim());
  const hasImage = Boolean(state.selectedFile);
  if (hasText && hasImage) {
    return "image_text";
  }
  if (hasImage) {
    return "image";
  }
  if (hasText) {
    return "text";
  }
  return null;
}

function modalityValidationError() {
  if (!state.serviceReady) {
    return {
      field: "prompt",
      message: "Service not ready.",
    };
  }
  const modality = currentInputModality();
  if (modality === null) {
    return {
      field: "prompt",
      message: "Add text or image.",
    };
  }
  if (
    state.supportedModalities === null ||
    state.supportedModalities.has(modality)
  ) {
    return null;
  }
  if (modality === "text" && state.supportedModalities.has("image_text")) {
    return {
      field: "image",
      message: "Add an image; this target requires text + image.",
    };
  }
  if (modality === "image" && state.supportedModalities.has("image_text")) {
    return {
      field: "prompt",
      message: "Add text; image-only is unsupported.",
    };
  }
  return {
    field: modality === "text" ? "prompt" : "image",
    message: `Supported: ${formatInputModalities(Array.from(state.supportedModalities))}.`,
  };
}

function updateEvaluateAvailability() {
  const validationError = modalityValidationError();
  elements.evaluateButton.disabled =
    state.evaluating || validationError !== null;
  elements.evaluateButton.title =
    state.evaluating || validationError === null ? "" : validationError.message;
}

function showModalityValidationError(validationError) {
  if (validationError.field === "image") {
    elements.imageError.textContent = validationError.message;
    elements.dropZone.focus();
  } else {
    elements.promptError.textContent = validationError.message;
    elements.promptText.focus();
  }
}

async function submitEvaluation(event) {
  event.preventDefault();
  clearFormErrors();

  const text = elements.promptText.value;
  const modalityError = modalityValidationError();
  if (modalityError !== null) {
    showModalityValidationError(modalityError);
    return;
  }
  if (text.length > state.textLimit) {
    elements.promptError.textContent =
      `Limit: ${formatInteger(state.textLimit)} characters.`;
    elements.promptText.focus();
    return;
  }

  setEvaluateLoading(true);
  renderResultLoading();

  try {
    const requestPayload = { text };
    if (state.selectedFile) {
      requestPayload.image = {
        name: state.selectedFile.name,
        media_type: state.selectedFile.type,
        base64: await readFileAsBase64(state.selectedFile),
      };
    }

    const response = await requestJson(API.evaluate, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });
    renderEvaluationResponse(response);
    showToast("Complete", "Decision ready.", "success");
    state.historyOffset = 0;
    state.historyDirty = true;
  } catch (error) {
    renderResultError("Evaluation failed", readableError(error));
    showToast("Evaluation failed", readableError(error), "error");
  } finally {
    setEvaluateLoading(false);
  }
}

function clearFormErrors() {
  elements.promptError.textContent = "";
  elements.imageError.textContent = "";
}

function setEvaluateLoading(loading) {
  state.evaluating = loading;
  elements.evaluateButton.classList.toggle("is-loading", loading);
  elements.evaluateButton.setAttribute("aria-busy", String(loading));
  elements.buttonLabel.textContent = loading ? "Evaluating…" : "Evaluate";
  updateEvaluateAvailability();
}

function renderResultLoading() {
  clearNode(elements.resultContainer);
  elements.resultState.textContent = "Analyzing";
  elements.resultState.dataset.tone = "review";

  const wrapper = createElement("div", "result-loading");
  wrapper.setAttribute("role", "status");
  wrapper.append(createElement("span", "", ""));
  wrapper.append(createElement("strong", "", "Evaluating"));
  wrapper.append(
    createElement("small", "", "Running active target."),
  );
  elements.resultContainer.append(wrapper);
}

function renderEvaluationResponse(response) {
  const result = response.result ?? response;
  const view = decisionView(result, {
    durationMs: response.duration_ms,
    historyId: response.history_id,
  });
  renderDecision(elements.resultContainer, view);
  elements.resultState.textContent = humanize(view.assessment);
  elements.resultState.dataset.tone = toneFor(view.assessment);
}

function renderDecision(container, view, options = {}) {
  clearNode(container);
  const wrapper = createElement("div", "decision-display");

  const banner = createElement("section", "decision-banner");
  banner.dataset.tone = toneFor(view.assessment);
  banner.append(createElement("p", "decision-banner-label", "Decision"));
  banner.append(
    createElement("strong", "decision-banner-value", humanize(view.assessment)),
  );
  banner.append(
    createElement(
      "p",
      "decision-banner-note",
      decisionSummary(view),
    ),
  );
  wrapper.append(banner);

  const comparison = createElement("div", "decision-comparison");
  comparison.append(
    createComparisonCell("Classifier", view.verdict),
    createComparisonCell("Effective", view.effectiveAction),
    createComparisonCell("Mode", view.trafficMode),
    createComparisonCell("Uncertain", view.uncertain ? "Yes" : "No"),
  );
  wrapper.append(comparison);

  const riskPanel = createElement("section", "risk-panel");
  const riskHeading = createElement("div", "risk-heading");
  riskHeading.append(createElement("span", "", "Risk"));
  riskHeading.append(
    createElement(
      "strong",
      "",
      view.riskScore === null ? "Unknown" : formatPercent(view.riskScore),
    ),
  );
  riskPanel.append(riskHeading);

  const progress = document.createElement("progress");
  progress.max = 1;
  progress.value = view.riskScore === null ? 0 : clamp(view.riskScore, 0, 1);
  progress.setAttribute(
    "aria-label",
    view.riskScore === null
      ? "Risk unknown"
      : `Risk ${formatPercent(view.riskScore)}`,
  );
  riskPanel.append(progress);

  const thresholdCopy = createElement("p", "risk-thresholds");
  thresholdCopy.append(
    createElement(
      "span",
      "",
      view.reviewThreshold === null
        ? "Review: adaptive"
        : `Review: ${formatPercent(view.reviewThreshold)}`,
    ),
    createElement(
      "span",
      "",
      view.blockThreshold === null
        ? "Block: unknown"
        : `Block: ${formatPercent(view.blockThreshold)}`,
    ),
  );
  riskPanel.append(thresholdCopy);
  wrapper.append(riskPanel);

  if (view.headDecisions.length) {
    const heads = createElement("section", "head-decision-panel");
    heads.append(createElement("span", "", "Detector heads"));
    const grid = createElement("div", "head-decision-grid");
    view.headDecisions.forEach((head) => {
      const card = createElement("div", "head-decision-card");
      card.dataset.tone = toneFor(head.recommendedAction);
      const decisive = head.name === view.decisiveHead;
      card.dataset.decisive = decisive ? "true" : "false";
      card.append(
        createElement(
          "strong",
          "",
          `${humanize(head.name)} head${decisive ? " (decisive)" : ""}`,
        ),
        createElement(
          "span",
          "",
          head.riskScore === null ? "Risk unknown" : `Risk ${formatPercent(head.riskScore)}`,
        ),
        createElement("small", "", humanize(head.recommendedAction)),
      );
      grid.append(card);
    });
    heads.append(grid);
    wrapper.append(heads);
  }

  const reasons = createElement("section", "reason-panel");
  reasons.append(createElement("span", "", "Signals"));
  const chips = createElement("div", "chip-list");
  const reasonValues = view.reasons.length ? view.reasons : ["None"];
  reasonValues.forEach((reason) => {
    chips.append(createElement("span", "chip", humanize(reason)));
  });
  reasons.append(chips);
  wrapper.append(reasons);

  wrapper.append(
    createMetadataGrid([
      ["Modality", humanize(view.modality)],
      ["Model", view.model],
      ["Latency", formatDuration(view.durationMs)],
      ["Trace ID", view.traceId],
      ["History ID", view.historyId],
    ]),
  );

  if (options.compact) {
    wrapper.classList.add("is-compact");
  }
  container.append(wrapper);
}

function decisionSummary(view) {
  if (view.assessment === "error") {
    return "Classification failed.";
  }
  if (
    view.riskScore !== null &&
    view.blockThreshold !== null &&
    view.assessment === "block" &&
    view.riskScore >= view.blockThreshold
  ) {
    return `Risk ${formatPercent(view.riskScore)} meets block ${formatPercent(view.blockThreshold)}.`;
  }
  if (
    view.riskScore !== null &&
    view.blockThreshold !== null &&
    view.assessment === "review"
  ) {
    const comparison =
      view.riskScore < view.blockThreshold ? "is below" : "meets";
    return (
      `Risk ${formatPercent(view.riskScore)} ${comparison} block ` +
      `${formatPercent(view.blockThreshold)}; review recommended.`
    );
  }
  if (
    view.riskScore !== null &&
    view.reviewThreshold !== null &&
    view.assessment === "allow" &&
    view.riskScore < view.reviewThreshold
  ) {
    return `Risk ${formatPercent(view.riskScore)} is below review ${formatPercent(view.reviewThreshold)}.`;
  }
  if (view.assessment === "review") {
    return "Manual review recommended.";
  }
  if (view.assessment === "block") {
    return "Blocking recommended.";
  }
  if (view.assessment === "allow") {
    return "No intervention recommended.";
  }
  return "No decision available.";
}

function createComparisonCell(label, value) {
  const cell = createElement("div", "comparison-cell");
  cell.dataset.tone = toneFor(value);
  cell.append(createElement("span", "", label));
  cell.append(createElement("strong", "", humanize(value)));
  return cell;
}

function createMetadataGrid(entries, className = "metadata-grid") {
  const grid = createElement("div", className);
  entries.forEach(([label, value]) => {
    const cell = createElement("div", "metadata-cell");
    cell.append(createElement("span", "", label));
    const valueElement = createElement("strong", "", displayValue(value));
    valueElement.title = displayValue(value);
    cell.append(valueElement);
    grid.append(cell);
  });
  return grid;
}

function renderResultError(title, message) {
  clearNode(elements.resultContainer);
  elements.resultState.textContent = "Error";
  elements.resultState.dataset.tone = "error";
  const error = createElement("div", "result-error");
  error.setAttribute("role", "alert");
  error.append(createElement("strong", "", title));
  error.append(createElement("p", "", message));
  elements.resultContainer.append(error);
}

function decisionView(result, additions = {}) {
  const decisions = Array.isArray(result?.decisions) ? result.decisions : [];
  const decision = decisions[0] ?? result?.decision ?? result ?? {};
  const summary = result?.summary ?? {};
  const effectiveAction =
    decision.enforcement_action ??
    decision.effective_action ??
    decision.action ??
    summary.action ??
    "unknown";
  const recommendedAction =
    decision.recommended_action ??
    decision.recommendation ??
    summary.recommended_action ??
    effectiveAction;
  const verdict = String(decision.verdict ?? result?.verdict ?? "unknown");
  const headDecisions = Array.isArray(decision.head_decisions)
    ? decision.head_decisions
        .filter((head) => head && typeof head === "object")
        .map((head) => ({
          name: String(head.name ?? "unknown"),
          riskScore: toFiniteNumber(head.risk_score),
          recommendedAction: String(head.recommended_action ?? "unknown"),
          threshold: toFiniteNumber(head.threshold),
          reviewThreshold: toFiniteNumber(head.review_threshold),
        }))
    : [];

  return {
    effectiveAction: String(effectiveAction),
    recommendedAction: String(recommendedAction),
    verdict,
    decisiveHead:
      typeof decision.decisive_head === "string" ? decision.decisive_head : null,
    headDecisions,
    assessment: primaryAssessment(verdict, recommendedAction),
    riskScore: toFiniteNumber(decision.risk_score ?? result?.risk_score),
    blockThreshold: toFiniteNumber(
      decision.threshold ?? decision.block_threshold ?? result?.threshold,
    ),
    reviewThreshold: toFiniteNumber(
      decision.review_threshold ?? result?.review_threshold,
    ),
    uncertain: Boolean(decision.uncertain ?? result?.uncertain),
    reasons: toStringArray(decision.reasons ?? result?.reasons),
    trafficMode: String(
      decision.traffic_mode ?? summary.traffic_mode ?? result?.traffic_mode ?? "unknown",
    ),
    modality: String(decision.modality ?? result?.modality ?? "unknown"),
    model: String(
      decision.model_id ??
        result?.model_id ??
      decision.model_family ??
        result?.model_family ??
        "—",
    ),
    traceId: String(
      result?.trace_id ?? decision.trace_id ?? additions.traceId ?? "—",
    ),
    durationMs: firstNumber(
      additions.durationMs,
      result?.duration_ms,
      decision.duration_ms,
      result?.latency_ms,
    ),
    historyId: String(
      additions.historyId ?? result?.history_id ?? decision.history_id ?? "—",
    ),
  };
}

function primaryAssessment(verdict, recommendedAction) {
  const normalizedVerdict = String(verdict ?? "").trim().toLowerCase();
  const normalizedAction = String(recommendedAction ?? "").trim().toLowerCase();
  if (normalizedVerdict === "guardrail_error") {
    return "error";
  }
  if (["allow", "review", "block"].includes(normalizedAction)) {
    return normalizedAction;
  }
  if (normalizedVerdict === "malicious") {
    return "block";
  }
  if (normalizedVerdict === "benign") {
    return "allow";
  }
  return "unknown";
}

function applyFilters(event) {
  if (event?.preventDefault) {
    event.preventDefault();
  }
  clearTimeout(state.filterTimer);
  state.historyOffset = 0;
  state.historyLimit = Number.parseInt(elements.filterLimit.value, 10) || 25;
  loadHistory();
}

function scheduleFilterUpdate() {
  clearTimeout(state.filterTimer);
  state.filterTimer = setTimeout(() => {
    state.historyOffset = 0;
    loadHistory();
  }, 320);
}

function resetFilters() {
  elements.historyFilters.reset();
  elements.filterLimit.value = "25";
  state.historyOffset = 0;
  state.historyLimit = 25;
  loadHistory();
  elements.historySearch.focus();
}

async function loadHistory() {
  const requestId = ++state.historyRequest;
  setHistoryLoading();

  const params = new URLSearchParams();
  appendFilter(params, "q", elements.historySearch.value.trim());
  appendFilter(params, "action", elements.filterAction.value);
  appendFilter(params, "verdict", elements.filterVerdict.value);
  appendFilter(params, "modality", elements.filterModality.value);
  appendFilter(params, "status", elements.filterStatus.value);
  appendFilter(params, "from", elements.filterFrom.value);
  appendFilter(params, "to", elements.filterTo.value);
  params.set("limit", String(state.historyLimit));
  params.set("offset", String(state.historyOffset));

  try {
    const payload = await requestJson(`${API.history}?${params.toString()}`);
    if (requestId !== state.historyRequest) {
      return;
    }
    const records = normalizeHistoryRecords(payload);
    state.historyTotal =
      firstNumber(payload.total, payload.total_count, payload.count, records.length) ??
      records.length;
    state.historyLoaded = true;
    state.historyDirty = false;
    renderHistory(records);
  } catch (error) {
    if (requestId !== state.historyRequest) {
      return;
    }
    state.historyLoaded = true;
    state.historyDirty = true;
    renderHistoryError(readableError(error));
  }
}

function normalizeHistoryRecords(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }
  for (const key of ["items", "history", "records", "results"]) {
    if (Array.isArray(payload?.[key])) {
      return payload[key];
    }
  }
  return [];
}

function renderHistory(records) {
  clearNode(elements.historyBody);
  hideHistoryMessage();

  if (!records.length) {
    showHistoryMessage(
      "No results",
      hasActiveFilters()
        ? "Clear filters to widen results."
        : "Runs appear here.",
    );
  } else {
    records.forEach((record) => elements.historyBody.append(createHistoryRow(record)));
  }

  const shownStart = state.historyTotal === 0 ? 0 : state.historyOffset + 1;
  const shownEnd = Math.min(state.historyOffset + records.length, state.historyTotal);
  elements.historyCount.textContent =
    `${formatInteger(shownStart)}–${formatInteger(shownEnd)} / ${formatInteger(state.historyTotal)}`;
  elements.activeFilterNote.textContent = activeFilterDescription();
  updatePagination(records.length);
}

function createHistoryRow(record) {
  const view = historyRecordView(record);
  const row = document.createElement("tr");

  const timeCell = document.createElement("td");
  const time = createElement("div", "history-time");
  const dateParts = formatDateParts(view.createdAt);
  time.append(createElement("strong", "", dateParts.date));
  time.append(createElement("small", "", dateParts.time));
  timeCell.append(time);
  row.append(timeCell);

  const promptCell = document.createElement("td");
  const prompt = createElement("div", "prompt-preview");
  prompt.append(
    createElement(
      "strong",
      "",
      view.text.trim() || (view.hasImage ? "Image only" : "Unavailable"),
    ),
  );
  prompt.append(createElement("small", "", view.requestId));
  promptCell.append(prompt);
  row.append(promptCell);

  row.append(badgeCell(view.modality), badgeCell(view.verdict));

  const riskCell = document.createElement("td");
  riskCell.append(
    createElement(
      "span",
      "risk-value",
      view.riskScore === null ? "—" : formatPercent(view.riskScore),
    ),
  );
  row.append(riskCell);
  row.append(badgeCell(view.recommendedAction), badgeCell(view.effectiveAction));
  row.append(badgeCell(view.status));

  const detailCell = document.createElement("td");
  const detailButton = createElement("button", "row-detail-button", "→");
  detailButton.type = "button";
  detailButton.setAttribute(
    "aria-label",
    `View details for ${view.requestId === "—" ? "evaluation" : view.requestId}`,
  );
  detailButton.addEventListener("click", () => openHistoryDialog(view.id));
  detailCell.append(detailButton);
  row.append(detailCell);

  return row;
}

function badgeCell(value) {
  const cell = document.createElement("td");
  const badge = createElement("span", "badge", humanize(value));
  badge.dataset.tone = toneFor(value);
  cell.append(badge);
  return cell;
}

function historyRecordView(record) {
  const result = record.result ?? record.response ?? record.output ?? record;
  const decision = decisionView(result, {
    durationMs: record.duration_ms ?? record.latency_ms,
    historyId: record.id ?? record.history_id,
    traceId: record.trace_id,
  });
  const rawStatus =
    record.status ??
    record.state ??
    (record.error || record.error_message ? "error" : "completed");

  return {
    ...decision,
    id: String(record.id ?? record.history_id ?? decision.historyId),
    createdAt:
      record.created_at ??
      record.timestamp ??
      record.timestamp_utc ??
      record.evaluated_at ??
      "",
    text: String(
      record.input?.text ??
        record.text_preview ??
        record.text ??
        record.prompt ??
        "",
    ),
    hasImage: Boolean(
      record.has_image ??
        record.image_count ??
        record.image_name ??
        record.image_url ??
        record.input?.image ??
        record.input?.images?.length ??
        record.images?.length,
    ),
    requestId: String(
      record.request_id ??
        record.input?.request_id ??
        firstDecision(result)?.request_id ??
        "—",
    ),
    status: String(rawStatus),
  };
}

function setHistoryLoading() {
  clearNode(elements.historyBody);
  showHistoryMessage("Loading", "Fetching records…", true);
  elements.historyCount.textContent = "Loading…";
  elements.activeFilterNote.textContent = "";
  elements.previousPage.disabled = true;
  elements.nextPage.disabled = true;
  elements.clearHistory.disabled = true;
}

function renderHistoryError(message) {
  clearNode(elements.historyBody);
  showHistoryMessage("Unavailable", message);
  elements.historyCount.textContent = "Load failed";
  elements.activeFilterNote.textContent = "";
  elements.previousPage.disabled = true;
  elements.nextPage.disabled = true;
  elements.clearHistory.disabled = false;
}

function showHistoryMessage(title, detail, loading = false) {
  clearNode(elements.historyMessage);
  elements.historyMessage.hidden = false;
  elements.historyMessage.classList.toggle("is-loading", loading);
  const copy = createElement("div");
  copy.append(createElement("strong", "", title));
  copy.append(createElement("span", "", detail));
  elements.historyMessage.append(copy);
}

function hideHistoryMessage() {
  elements.historyMessage.hidden = true;
  elements.historyMessage.classList.remove("is-loading");
  clearNode(elements.historyMessage);
}

function updatePagination(recordsOnPage) {
  const page = Math.floor(state.historyOffset / state.historyLimit) + 1;
  const totalPages = Math.max(1, Math.ceil(state.historyTotal / state.historyLimit));
  elements.pageStatus.textContent = `${formatInteger(page)} / ${formatInteger(totalPages)}`;
  elements.previousPage.disabled = state.historyOffset <= 0;
  elements.nextPage.disabled =
    recordsOnPage < state.historyLimit ||
    state.historyOffset + recordsOnPage >= state.historyTotal;
  elements.clearHistory.disabled = state.clearingHistory || state.historyTotal === 0;
}

function previousPage() {
  state.historyOffset = Math.max(0, state.historyOffset - state.historyLimit);
  loadHistory();
  scrollHistoryIntoView();
}

function nextPage() {
  state.historyOffset += state.historyLimit;
  loadHistory();
  scrollHistoryIntoView();
}

function scrollHistoryIntoView() {
  document.querySelector("#history-title").scrollIntoView({ behavior: "smooth", block: "start" });
}

function hasActiveFilters() {
  return Boolean(
    elements.historySearch.value.trim() ||
      elements.filterAction.value ||
      elements.filterVerdict.value ||
      elements.filterModality.value ||
      elements.filterStatus.value ||
      elements.filterFrom.value ||
      elements.filterTo.value,
  );
}

function activeFilterDescription() {
  const labels = [];
  if (elements.historySearch.value.trim()) {
    labels.push(`search “${elements.historySearch.value.trim()}”`);
  }
  if (elements.filterAction.value) {
    labels.push(`action: ${humanize(elements.filterAction.value)}`);
  }
  if (elements.filterVerdict.value) {
    labels.push(`classifier: ${humanize(elements.filterVerdict.value)}`);
  }
  if (elements.filterModality.value) {
    labels.push(`modality: ${humanize(elements.filterModality.value)}`);
  }
  if (elements.filterStatus.value) {
    labels.push(`status: ${humanize(elements.filterStatus.value)}`);
  }
  if (elements.filterFrom.value) {
    labels.push(`from ${elements.filterFrom.value}`);
  }
  if (elements.filterTo.value) {
    labels.push(`to ${elements.filterTo.value}`);
  }
  return labels.length ? labels.join(" · ") : "All records";
}

async function openHistoryDialog(id) {
  if (!id || id === "—" || id === "Not recorded" || id === "undefined") {
    showToast("Unavailable", "Missing history ID.", "error");
    return;
  }
  showDialog();
  renderDialogLoading();

  try {
    const payload = await requestJson(`${API.history}/${encodeURIComponent(id)}`);
    renderHistoryDetail(payload.item ?? payload.record ?? payload);
  } catch (error) {
    renderDialogError(readableError(error));
  }
}

function showDialog() {
  if (typeof elements.historyDialog.showModal === "function") {
    if (!elements.historyDialog.open) {
      elements.historyDialog.showModal();
    }
  } else {
    elements.historyDialog.setAttribute("open", "");
  }
}

function closeHistoryDialog() {
  releaseHistoryImageUrls();
  if (typeof elements.historyDialog.close === "function") {
    elements.historyDialog.close();
  } else {
    elements.historyDialog.removeAttribute("open");
  }
}

function handleDialogBackdropClick(event) {
  if (event.target === elements.historyDialog) {
    closeHistoryDialog();
  }
}

function openClearHistoryDialog() {
  elements.clearHistoryError.textContent = "";
  state.dialogReturnFocus = elements.clearHistory;
  showConfirmationDialog(elements.clearHistoryDialog);
  elements.cancelClearHistory.focus();
}

function closeClearHistoryDialog() {
  if (state.clearingHistory) {
    return;
  }
  closeConfirmationDialog(elements.clearHistoryDialog);
}

async function confirmHistoryClear() {
  if (state.clearingHistory) {
    return;
  }
  elements.clearHistoryError.textContent = "";
  setHistoryClearLoading(true);
  try {
    const payload = await requestJson(API.history, {
      method: "DELETE",
      headers: {
        "X-AEGIS-Confirmation": "clear-history",
      },
    });
    const deletedRecords =
      firstNumber(payload.deleted_records, payload.cleared, state.historyTotal) ?? 0;
    state.historyOffset = 0;
    state.historyTotal = 0;
    state.historyDirty = true;
    closeConfirmationDialog(elements.clearHistoryDialog);
    await Promise.allSettled([loadSummary(), loadHistory()]);
    showToast(
      "Cleared",
      deletedRecords === 1
        ? "1 record and its images deleted."
        : `${formatInteger(deletedRecords)} records and their images deleted.`,
      "success",
    );
  } catch (error) {
    elements.clearHistoryError.textContent = readableError(error);
  } finally {
    setHistoryClearLoading(false);
  }
}

function setHistoryClearLoading(loading) {
  state.clearingHistory = loading;
  elements.clearHistoryDialog.setAttribute("aria-busy", String(loading));
  elements.cancelClearHistory.disabled = loading;
  elements.confirmClearHistory.disabled = loading;
  elements.confirmClearHistory.querySelector(".button-label").textContent =
    loading ? "Clearing…" : "Clear";
  elements.clearHistory.disabled = loading || state.historyTotal === 0;
}

function handleConfirmBackdropClick(event) {
  if (event.target !== event.currentTarget) {
    return;
  }
  if (event.currentTarget === elements.clearHistoryDialog) {
    closeClearHistoryDialog();
  } else if (event.currentTarget === elements.enforceDialog) {
    closeEnforceDialog();
  }
}

function showConfirmationDialog(dialog) {
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) {
      dialog.showModal();
    }
  } else {
    dialog.setAttribute("open", "");
  }
}

function closeConfirmationDialog(dialog) {
  if (typeof dialog.close === "function") {
    if (dialog.open) {
      dialog.close();
    }
  } else {
    dialog.removeAttribute("open");
    restoreDialogFocus();
  }
}

function restoreDialogFocus() {
  const target = state.dialogReturnFocus;
  state.dialogReturnFocus = null;
  if (target && target.isConnected && !target.disabled) {
    target.focus();
    return;
  }
  document.querySelector(`#${ROUTES[state.activeRoute].heading}`)?.focus({
    preventScroll: true,
  });
}

function renderDialogLoading() {
  clearNode(elements.dialogContent);
  const loading = createElement("div", "dialog-loading");
  const copy = createElement("div");
  copy.append(createElement("span"));
  copy.append(createElement("p", "", "Loading…"));
  loading.append(copy);
  elements.dialogContent.append(loading);
}

function renderDialogError(message) {
  clearNode(elements.dialogContent);
  const error = createElement("div", "result-error");
  error.setAttribute("role", "alert");
  error.append(createElement("strong", "", "Load failed"));
  error.append(createElement("p", "", message));
  elements.dialogContent.append(error);
}

function renderHistoryDetail(record) {
  releaseHistoryImageUrls();
  const imageRequest = state.historyImageRequest;
  clearNode(elements.dialogContent);
  const view = historyRecordView(record);
  const layout = createElement("div", "detail-layout");

  const promptSection = createElement("section", "detail-prompt");
  promptSection.append(createElement("span", "detail-label", "Prompt"));
  promptSection.append(
    createElement(
      "p",
      "",
      view.text.trim() || (view.hasImage ? "Image only" : "No text."),
    ),
  );
  layout.append(promptSection);

  if (record.error && typeof record.error === "object") {
    const errorSection = createElement("section", "result-error detail-json-error");
    errorSection.setAttribute("role", "alert");
    errorSection.append(createElement("strong", "", "Error"));
    errorSection.append(
      createElement(
        "p",
        "",
        String(
          record.error.message ??
            record.error.detail ??
            record.error.code ??
            "Evaluation failed.",
        ),
      ),
    );
    layout.append(errorSection);
  }

  const images = collectImageSources(record);
  if (images.length) {
    const imageSection = createElement("section", "detail-images");
    imageSection.append(createElement("span", "detail-label", "Image"));
    const imageGrid = createElement("div", "detail-image-grid");
    images.forEach((imageRecord, index) => {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.alt = imageRecord.name
        ? `Submitted image: ${imageRecord.name}`
        : `Submitted image ${index + 1}`;
      image.loading = "lazy";
      if (imageRecord.url.startsWith("data:") || imageRecord.url.startsWith("blob:")) {
        image.src = imageRecord.url;
      } else {
        loadProtectedHistoryImage(image, imageRecord.url, imageRequest);
      }
      figure.append(image);
      figure.append(
        createElement("figcaption", "", imageRecord.name || `Image ${index + 1}`),
      );
      imageGrid.append(figure);
    });
    imageSection.append(imageGrid);
    layout.append(imageSection);
  }

  layout.append(
    createMetadataGrid(
      [
        ["Recorded", formatDateParts(view.createdAt).combined],
        ["Request ID", view.requestId],
        ["Status", humanize(view.status)],
        ["Modality", humanize(view.modality)],
        ["Latency", formatDuration(view.durationMs)],
        ["History ID", view.id],
      ],
      "detail-metadata",
    ),
  );

  const dialogResult = createElement("section", "dialog-result");
  dialogResult.append(createElement("span", "detail-label", "Decision"));
  renderDecision(dialogResult, view, { compact: true });
  layout.append(dialogResult);

  const rawPayload =
    record.output ?? record.response ?? record.result ?? record.error ?? null;
  if (rawPayload && typeof rawPayload === "object") {
    const rawDetails = createElement("details", "raw-output");
    rawDetails.append(
      createElement(
        "summary",
        "",
        record.output || record.response || record.result
          ? "JSON"
          : "Error JSON",
      ),
    );
    rawDetails.append(
      createElement("pre", "", JSON.stringify(rawPayload, null, 2)),
    );
    layout.append(rawDetails);
  }

  elements.dialogContent.append(layout);
}

function releaseHistoryImageUrls() {
  state.historyImageRequest += 1;
  state.historyImageUrls.forEach((url) => URL.revokeObjectURL(url));
  state.historyImageUrls = [];
}

async function loadProtectedHistoryImage(image, url, requestId) {
  try {
    const response = await requestApi(url, {
      headers: { Accept: "image/png,image/jpeg,image/webp" },
    });
    if (!response.ok) {
      throw new Error(`Image request failed (${response.status}).`);
    }
    const contentType = response.headers.get("Content-Type")?.split(";", 1)[0] ?? "";
    if (!ALLOWED_IMAGE_TYPES.has(contentType)) {
      throw new Error("The history image response had an invalid media type.");
    }
    const objectUrl = URL.createObjectURL(await response.blob());
    if (requestId !== state.historyImageRequest || !image.isConnected) {
      URL.revokeObjectURL(objectUrl);
      return;
    }
    state.historyImageUrls.push(objectUrl);
    image.src = objectUrl;
  } catch {
    image.alt = `${image.alt} (unavailable)`;
  }
}

function collectImageSources(record) {
  const candidates = [];
  if (record.image_url) {
    candidates.push({ url: record.image_url, name: record.image_name });
  }
  if (Array.isArray(record.image_urls)) {
    record.image_urls.forEach((url) => candidates.push({ url }));
  }
  if (Array.isArray(record.images)) {
    record.images.forEach((image) => {
      if (typeof image === "string") {
        candidates.push({ url: image });
      } else if (image && typeof image === "object") {
        candidates.push({
          url: image.url ?? image.image_url ?? image.src,
          name: image.name ?? image.filename,
        });
      }
    });
  }
  if (record.input?.image) {
    const image = record.input.image;
    candidates.push({
      url: image.url ?? image.image_url ?? image.src,
      name: image.name ?? image.filename,
    });
  }

  return candidates
    .map((candidate) => ({
      url: safeImageUrl(candidate.url),
      name: candidate.name ? String(candidate.name) : "",
    }))
    .filter((candidate) => Boolean(candidate.url));
}

function safeImageUrl(value) {
  if (!value || typeof value !== "string") {
    return "";
  }
  if (/^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=\s]+$/i.test(value)) {
    return value;
  }
  if (value.startsWith("blob:")) {
    return value;
  }
  try {
    const parsed = new URL(value, window.location.href);
    return parsed.origin === window.location.origin ? parsed.href : "";
  } catch {
    return "";
  }
}

async function requestJson(url, options = {}) {
  const response = await requestApi(url, options);

  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const message =
      payload?.error?.message ??
      payload?.message ??
      payload?.detail ??
      `Request failed (${response.status}).`;
    throw new Error(String(message));
  }
  if (payload === null || typeof payload !== "object") {
    throw new Error("Invalid server response.");
  }
  return payload;
}

async function requestApi(url, options = {}) {
  if (!GUI_SESSION_PATTERN.test(guiSessionToken)) {
    throw new Error("GUI session unavailable. Reopen the per-launch dashboard URL.");
  }
  const parsed = new URL(url, window.location.origin);
  if (parsed.origin !== window.location.origin || !parsed.pathname.startsWith("/api/")) {
    throw new Error("Refusing to send the GUI session token outside the local API.");
  }
  const headers = new Headers(options.headers ?? {});
  headers.set(GUI_SESSION_HEADER, guiSessionToken);
  return fetch(parsed.href, {
    cache: "no-store",
    credentials: "omit",
    ...options,
    headers,
  });
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const result = String(reader.result ?? "");
      const comma = result.indexOf(",");
      if (comma < 0) {
        reject(new Error("Image encoding failed. Try another file."));
        return;
      }
      resolve(result.slice(comma + 1));
    });
    reader.addEventListener("error", () => {
      reject(new Error("Image read failed. Re-select it."));
    });
    reader.readAsDataURL(file);
  });
}

function appendFilter(params, name, value) {
  if (value) {
    params.set(name, value);
  }
}

function firstDecision(result) {
  return Array.isArray(result?.decisions) ? result.decisions[0] ?? {} : {};
}

function createElement(tagName, className = "", text = null) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== null && text !== undefined) {
    element.textContent = String(text);
  }
  return element;
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function humanize(value) {
  if (value === null || value === undefined || value === "") {
    return "Unknown";
  }
  if (String(value).toLowerCase() === "image_text") {
    return "Multimodal";
  }
  return String(value)
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  return String(value);
}

function toneFor(value) {
  const normalized = String(value ?? "").toLowerCase();
  if (["allow", "allowed", "benign", "completed", "online", "yes"].includes(normalized)) {
    return normalized === "allowed" ? "allow" : normalized;
  }
  if (["review", "unknown", "pending"].includes(normalized)) {
    return normalized;
  }
  if (
    ["block", "blocked", "malicious", "error", "guardrail_error", "offline", "no"].includes(
      normalized,
    )
  ) {
    if (normalized === "blocked") {
      return "block";
    }
    return normalized;
  }
  return "neutral";
}

function toStringArray(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (value === null || value === undefined || value === "") {
    return [];
  }
  return [String(value)];
}

function toFiniteNumber(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function firstNumber(...values) {
  for (const value of values) {
    const number = toFiniteNumber(value);
    if (number !== null) {
      return number;
    }
  }
  return null;
}

function sumCounts(counts) {
  if (!counts || typeof counts !== "object") {
    return null;
  }
  let total = 0;
  let found = false;
  Object.values(counts).forEach((value) => {
    const number = toFiniteNumber(value);
    if (number !== null) {
      total += number;
      found = true;
    }
  });
  return found ? total : null;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatInteger(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

function formatPercent(value) {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    minimumFractionDigits: value > 0 && value < 0.01 ? 1 : 0,
    maximumFractionDigits: 1,
  }).format(value);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 bytes";
  }
  const units = ["bytes", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${new Intl.NumberFormat(undefined, {
    maximumFractionDigits: index === 0 ? 0 : 1,
  }).format(value)} ${units[index]}`;
}

function humanizeMediaType(mediaType) {
  const labels = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WebP",
  };
  return labels[mediaType] ?? humanize(mediaType);
}

function formatDuration(milliseconds) {
  const value = toFiniteNumber(milliseconds);
  if (value === null) {
    return "—";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)} s`;
}

function formatDateParts(value) {
  if (!value) {
    return { date: "Unknown", time: "", combined: "—" };
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return { date: String(value), time: "", combined: String(value) };
  }
  const dateText = new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(date);
  const timeText = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
  return { date: dateText, time: timeText, combined: `${dateText}, ${timeText}` };
}

function readableError(error) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Unexpected error.";
}

function showToast(title, message, tone = "info") {
  const toast = createElement("div", "toast");
  toast.dataset.tone = tone;
  toast.setAttribute("role", tone === "error" ? "alert" : "status");
  toast.append(createElement("strong", "", title));
  toast.append(createElement("span", "", message));
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 4600);
}
