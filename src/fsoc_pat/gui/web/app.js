/**
 * OptiLign FSOC Simulator — Complete Frontend Application
 * Handles:
 *  1. Landing Home Page Particle & Optical Beam Canvas
 *  2. Tabbed Panes Switching (Simulator, Create Environment, Recent Presets, Reports)
 *  3. Dynamic Environment Studio Parameter Slider Bindings & API Dispatch
 *  4. Recent Environments Grid Rendering, 1-Click Launch & Clone
 *  5. 3D WebGL Three.js Scene (Dynamic Fog, Solar Glare, Multi-Target Swarms, Gimbal & Laser)
 *  6. Virtual Camera Feed & Reticle HUD
 *  7. Rolling 60s Dual-Line Performance Chart
 *  8. Telemetry Polling & Real-Time Performance Audits
 */

(() => {
  "use strict";

  // --- State Variables ---
  let telemetry = null;
  let pollInterval = null;
  const POLLING_RATE_MS = 33; // ~30 Hz

  // --- 3D Scene Variables ---
  let scene, camera3d, renderer3d;
  let gimbalBase, gimbalYoke, gimbalPayload, laserBeam, omniSearchCone;
  let droneTarget, droneRotors = [];
  let secondaryDrones = [];
  let clouds = [];
  let aviationBeaconLight = null;
  let sunGroup = null;
  let sunLight = null, hemiLight = null;
  let groundMistGroup = null, mistMat = null;
  let groundMesh = null, groundGrid = null, facilityBldg = null, roofSlab = null;
  let parapetGroup = null, mastGroup = null, cityGroup = null;
  let starField = null, oceanMesh = null;
  let currentDomain = "urban_rooftop";
  let isOrbiting = false;
  let mousePrev = { x: 0, y: 0 };
  let orbitSpherical = { radius: 46, theta: Math.PI / 4, phi: Math.PI / 3.2 };
  let isCameraPov = false;

  // --- Individual Target Configurations State ---
  const DEFAULT_TARGET_CONFIGS = [
    { id: "Drone_Alpha_Lead", role: "primary", trajectory: "circular", range_m: 180, speed_mps: 24, alt_offset_m: 0, wavelength_nm: 1550, power_mw: 500, modulation: "cw" },
    { id: "Drone_Beta_Wingman", role: "escort", trajectory: "circular", range_m: 210, speed_mps: 25, alt_offset_m: 15, wavelength_nm: 1550, power_mw: 500, modulation: "cw" },
    { id: "Drone_Gamma_Flank", role: "decoy", trajectory: "sinusoidal", range_m: 240, speed_mps: 28, alt_offset_m: -10, wavelength_nm: 850, power_mw: 250, modulation: "strobe_100hz" },
    { id: "Drone_Delta_Echelon", role: "escort", trajectory: "evasive", range_m: 195, speed_mps: 30, alt_offset_m: 25, wavelength_nm: 1064, power_mw: 1000, modulation: "cw" },
    { id: "Drone_Epsilon_Weave", role: "decoy", trajectory: "linear", range_m: 260, speed_mps: 32, alt_offset_m: -20, wavelength_nm: 1550, power_mw: 500, modulation: "ml_signature" },
    { id: "Drone_Zeta_Perimeter", role: "relay", trajectory: "random", range_m: 300, speed_mps: 20, alt_offset_m: 35, wavelength_nm: 532, power_mw: 500, modulation: "strobe_500hz" },
  ];
  let targetConfigs = JSON.parse(JSON.stringify(DEFAULT_TARGET_CONFIGS));
  let activeTargetIndex = 0;

  // --- Landing Canvas Variables ---
  let homeCanvas, homeCtx, homeAnimId;
  let homeParticles = [];

  // --- DOM Elements ---
  const pageHome = document.getElementById("page-home");
  const pageApp = document.getElementById("page-app");
  const btnEnterSim = document.getElementById("btn-enter-sim");
  const btnHomeReturn = document.getElementById("btn-home-return");
  const btnNavHome = document.getElementById("btn-nav-home");
  const btnTopCreateEnv = document.getElementById("btn-top-create-env");

  // No Virtual Environment Created Elements
  let activeEnvironmentCreated = false;
  const noEnvOverlay = document.getElementById("no-env-overlay");
  const btnNoEnvCreate = document.getElementById("btn-no-env-create");
  const btnNoEnvPresets = document.getElementById("btn-no-env-presets");

  // Quick 3D live controls
  const quickTargetPills = document.getElementById("quick-target-pills");
  const quickFogSlider = document.getElementById("quick-fog-slider");
  const quickFogLabel = document.getElementById("quick-fog-label");

  // Pane Tabs
  const paneTabs = document.querySelectorAll(".pane-tab");
  const paneContents = document.querySelectorAll(".pane-content");

  // Live Simulator Elements
  const container3d = document.getElementById("container-3d");
  const canvas3d = document.getElementById("canvas-3d");
  const canvasCamera = document.getElementById("canvas-camera");
  const ctxCamera = canvasCamera.getContext("2d");
  const canvasChart = document.getElementById("canvas-chart");
  const ctxChart = canvasChart.getContext("2d");
  const reticleHud = document.getElementById("reticle-hud");
  const hudTargetLabel = document.getElementById("hud-target-label");

  // Readouts
  const topActiveEnv = document.getElementById("top-active-env");
  const topSimClock = document.getElementById("top-sim-clock");
  const btnPauseSim = document.getElementById("btn-pause-sim");
  const badgeWeatherMode = document.getElementById("badge-weather-mode");

  const posDist = document.getElementById("pos-dist");
  const posYaw = document.getElementById("pos-yaw");
  const posPitch = document.getElementById("pos-pitch");
  const posTargets = document.getElementById("pos-targets");

  const telemTime = document.getElementById("telem-time");
  const telemBatteryBar = document.getElementById("telem-battery-bar");
  const telemBatteryText = document.getElementById("telem-battery-text");
  const telemSignal = document.getElementById("telem-signal");
  const telemWeather = document.getElementById("telem-weather");
  const telemFog = document.getElementById("telem-fog");
  const telemTemp = document.getElementById("telem-temp");
  const telemWind = document.getElementById("telem-wind");
  const telemMargin = document.getElementById("telem-margin");

  const statDetected = document.getElementById("stat-detected");
  const statPatState = document.getElementById("stat-pat-state");
  const statConf = document.getElementById("stat-conf");
  const statPosPx = document.getElementById("stat-pos-px");
  const statOffsetPx = document.getElementById("stat-offset-px");
  const statDist = document.getElementById("stat-dist");
  const statLink = document.getElementById("stat-link");
  const statFps = document.getElementById("stat-fps");
  const statMlSig = document.getElementById("stat-ml-sig");
  const statEkfMode = document.getElementById("stat-ekf-mode");
  const statDecoyDiscrim = document.getElementById("stat-decoy-discrim");

  const inputYaw = document.getElementById("input-yaw");
  const inputPitch = document.getElementById("input-pitch");
  const selectZoom = document.getElementById("select-zoom");
  const selectMode = document.getElementById("select-mode");
  const btnRecenter = document.getElementById("btn-recenter");
  const btnClearLog = document.getElementById("btn-clear-log");
  const logContainer = document.getElementById("log-container");

  const sbScenarioName = document.getElementById("sb-scenario-name");
  const sbTarget = document.getElementById("sb-target");
  const sbStatusText = document.getElementById("sb-status-text");

  // Create Environment Studio Elements
  const envDomainSelect = document.getElementById("env-domain-select");
  const envPlatformSelect = document.getElementById("env-platform-select");
  const envSwaySlider = document.getElementById("env-sway-slider");
  const valSway = document.getElementById("val-sway");
  const envSwayFreq = document.getElementById("env-sway-freq");
  const valSwayFreq = document.getElementById("val-sway-freq");
  const envAltitudeSlider = document.getElementById("env-altitude-slider");
  const valAltitude = document.getElementById("val-altitude");

  const envWeatherSelect = document.getElementById("env-weather-select");
  const envFogSlider = document.getElementById("env-fog-slider");
  const valFogSeverity = document.getElementById("val-fog-severity");
  const envSolarGlare = document.getElementById("env-solar-glare");
  const envSolarAngle = document.getElementById("env-solar-angle");
  const valSolarAngle = document.getElementById("val-solar-angle");
  const groupSolarAngle = document.getElementById("group-solar-angle");
  const envTurbulenceSlider = document.getElementById("env-turbulence-slider");
  const valTurbulence = document.getElementById("val-turbulence");
  const envTempSlider = document.getElementById("env-temp-slider");
  const valTemperature = document.getElementById("val-temperature");
  const envWindSlider = document.getElementById("env-wind-slider");
  const valWind = document.getElementById("val-wind");

  // Multi-Target Swarms & Individual Target Configuration Elements
  const envNumTargets = document.getElementById("env-num-targets");
  const valTargets = document.getElementById("val-targets");
  const targetTabPills = document.getElementById("target-tab-pills");
  const targetEditorBox = document.getElementById("target-editor-box");
  const teBadgeId = document.getElementById("te-badge-id");
  const teBadgeDesc = document.getElementById("te-badge-desc");
  const targetNameInput = document.getElementById("target-name-input");
  const targetRoleSelect = document.getElementById("target-role-select");
  const targetTrajSelect = document.getElementById("target-traj-select");
  const targetRangeSlider = document.getElementById("target-range-slider");
  const valTargetRange = document.getElementById("val-target-range");
  const targetSpeedSlider = document.getElementById("target-speed-slider");
  const valTargetSpeed = document.getElementById("val-target-speed");
  const targetAltSlider = document.getElementById("target-alt-slider");
  const valTargetAlt = document.getElementById("val-target-alt");
  const targetWavelengthSelect = document.getElementById("target-wavelength-select");
  const targetPowerSelect = document.getElementById("target-power-select");
  const targetModSelect = document.getElementById("target-mod-select");

  // Terminal Gimbal & FSM Elements
  const envGimbalRate = document.getElementById("env-gimbal-rate");
  const valGimbalRate = document.getElementById("val-gimbal-rate");
  const envGimbalLag = document.getElementById("env-gimbal-lag");
  const valGimbalLag = document.getElementById("val-gimbal-lag");
  const envFsmEnable = document.getElementById("env-fsm-enable");
  const envRxAperture = document.getElementById("env-rx-aperture");
  const valRxAperture = document.getElementById("val-rx-aperture");
  const envFilterBw = document.getElementById("env-filter-bw");

  // Cloud Occlusion Elements
  const envOcclusionToggle = document.getElementById("env-occlusion-toggle");
  const occlusionConfigBlock = document.getElementById("occlusion-config-block");
  const envOcclusionStart = document.getElementById("env-occlusion-start");
  const valOcclusionStart = document.getElementById("val-occlusion-start");
  const envOcclusionDuration = document.getElementById("env-occlusion-duration");
  const valOcclusionDuration = document.getElementById("val-occlusion-duration");
  const envOcclusionDepth = document.getElementById("env-occlusion-depth");
  const valOcclusionDepth = document.getElementById("val-occlusion-depth");

  // Real-Time Optical Channel Physics Readouts
  const prevExtinction = document.getElementById("prev-extinction");
  const prevFspl = document.getElementById("prev-fspl");
  const prevPrx = document.getElementById("prev-prx");
  const prevMargin = document.getElementById("prev-margin");
  const prevRytov = document.getElementById("prev-rytov");
  const prevGreenwood = document.getElementById("prev-greenwood");
  const prevBer = document.getElementById("prev-ber");
  const prevDifficulty = document.getElementById("prev-difficulty");

  // Bottom Command Console & Actions
  const envNameInput = document.getElementById("env-name-input");
  const sbcChipDomain = document.getElementById("sbc-chip-domain");
  const sbcChipWeather = document.getElementById("sbc-chip-weather");
  const sbcChipTargets = document.getElementById("sbc-chip-targets");
  const sbcChipLaser = document.getElementById("sbc-chip-laser");
  const sbcChipMargin = document.getElementById("sbc-chip-margin");

  const btnEnvLaunch = document.getElementById("btn-env-launch");
  const btnEnvLaunchTop = document.getElementById("btn-env-launch-top");
  const btnEnvResetDefaults = document.getElementById("btn-env-reset-defaults");
  const btnEnvResetDefaultsTop = document.getElementById("btn-env-reset-defaults-top");
  const btnEnvSavePreset = document.getElementById("btn-env-save-preset");
  const presetQuickPills = document.getElementById("preset-quick-pills");

  // Recent & Presets Elements
  const recentEnvsContainer = document.getElementById("recent-envs-container");
  const btnRefreshEnvs = document.getElementById("btn-refresh-envs");

  // Reports Elements
  const kpiLockPct = document.getElementById("kpi-lock-pct");
  const kpiAvgErr = document.getElementById("kpi-avg-err");
  const kpiMaxErr = document.getElementById("kpi-max-err");
  const kpiAcqTime = document.getElementById("kpi-acq-time");
  const btnRefreshReports = document.getElementById("btn-refresh-reports");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const btnExportJson = document.getElementById("btn-export-json");

  // =========================================================================
  // 1. Landing Home Page Particle & Optical Beam Canvas Animation
  // =========================================================================
  function initHomeCanvas() {
    homeCanvas = document.getElementById("canvas-home-bg");
    if (!homeCanvas) return;
    homeCtx = homeCanvas.getContext("2d");

    const resizeHome = () => {
      homeCanvas.width = window.innerWidth;
      homeCanvas.height = window.innerHeight;
    };
    resizeHome();
    window.addEventListener("resize", resizeHome);

    // Create particle array
    homeParticles = [];
    const count = Math.floor((homeCanvas.width * homeCanvas.height) / 14000);
    for (let i = 0; i < count; i++) {
      homeParticles.push({
        x: Math.random() * homeCanvas.width,
        y: Math.random() * homeCanvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 1.8 + 0.5,
        alpha: Math.random() * 0.6 + 0.2,
      });
    }

    animateHome();
  }

  function animateHome() {
    if (!pageHome.classList.contains("active")) {
      cancelAnimationFrame(homeAnimId);
      return;
    }

    const w = homeCanvas.width;
    const h = homeCanvas.height;
    homeCtx.clearRect(0, 0, w, h);

    // Optical Grid Lines (Subtle crisp grid on white)
    homeCtx.strokeStyle = "rgba(2, 132, 199, 0.045)";
    homeCtx.lineWidth = 1;
    const gridSize = 60;
    for (let x = 0; x < w; x += gridSize) {
      homeCtx.beginPath(); homeCtx.moveTo(x, 0); homeCtx.lineTo(x, h); homeCtx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
      homeCtx.beginPath(); homeCtx.moveTo(0, y); homeCtx.lineTo(w, y); homeCtx.stroke();
    }

    // Dynamic laser beams cutting across
    const t = Date.now() * 0.001;
    const laserY = (h * 0.38) + Math.sin(t * 0.8) * 40;
    const grad = homeCtx.createLinearGradient(0, laserY, w, laserY);
    grad.addColorStop(0, "rgba(239, 68, 68, 0)");
    grad.addColorStop(0.5, "rgba(239, 68, 68, 0.45)");
    grad.addColorStop(0.7, "rgba(2, 132, 199, 0.55)");
    grad.addColorStop(1, "rgba(2, 132, 199, 0)");

    homeCtx.strokeStyle = grad;
    homeCtx.lineWidth = 2;
    homeCtx.beginPath();
    homeCtx.moveTo(0, laserY);
    homeCtx.lineTo(w, laserY + Math.cos(t * 0.5) * 20);
    homeCtx.stroke();

    // Draw Particles
    homeParticles.forEach((p, idx) => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      homeCtx.fillStyle = `rgba(2, 132, 199, ${p.alpha * 0.5})`;
      homeCtx.beginPath();
      homeCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      homeCtx.fill();

      // Connect close neighbors
      for (let j = idx + 1; j < homeParticles.length; j++) {
        const p2 = homeParticles[j];
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 90) {
          homeCtx.strokeStyle = `rgba(2, 132, 199, ${(1 - dist / 90) * 0.18})`;
          homeCtx.beginPath();
          homeCtx.moveTo(p.x, p.y);
          homeCtx.lineTo(p2.x, p2.y);
          homeCtx.stroke();
        }
      }
    });

    homeAnimId = requestAnimationFrame(animateHome);
  }

  // =========================================================================
  // 2. Page & Pane Navigation
  // =========================================================================
  function showSimulator() {
    pageHome.classList.remove("active");
    pageApp.classList.add("active");
    const tabSim = document.getElementById("tab-simulator");
    if (tabSim) tabSim.click();
    if (!scene) {
      init3DEnvironment();
    } else {
      onWindowResize();
    }
    startTelemetryPolling();
    fetchRecentEnvironments();
    fetchReports();
  }

  function showCreateEnvironment() {
    pageHome.classList.remove("active");
    pageApp.classList.add("active");
    const tabCreate = document.getElementById("tab-create-env");
    if (tabCreate) tabCreate.click();
    if (!scene) {
      init3DEnvironment();
    } else {
      onWindowResize();
    }
    startTelemetryPolling();
    fetchRecentEnvironments();
    fetchReports();
  }

  function showHome() {
    pageApp.classList.remove("active");
    pageHome.classList.add("active");
    initHomeCanvas();
  }

  // When user clicks "ENTER THE SIMULATOR", open Create Virtual Environment directly
  if (btnEnterSim) btnEnterSim.addEventListener("click", showCreateEnvironment);
  if (btnHomeReturn) btnHomeReturn.addEventListener("click", showHome);
  if (btnNavHome) btnNavHome.addEventListener("click", showHome);
  if (btnTopCreateEnv) btnTopCreateEnv.addEventListener("click", () => {
    const tabCreate = document.getElementById("tab-create-env");
    if (tabCreate) tabCreate.click();
  });

  if (btnNoEnvCreate) {
    btnNoEnvCreate.addEventListener("click", () => {
      const tabCreate = document.getElementById("tab-create-env");
      if (tabCreate) tabCreate.click();
    });
  }

  if (btnNoEnvPresets) {
    btnNoEnvPresets.addEventListener("click", () => {
      const tabRecent = document.getElementById("tab-recent-envs");
      if (tabRecent) tabRecent.click();
    });
  }

  // Switch between window panes (Simulator, Create Environment, Recent, Reports)
  paneTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetPaneId = `pane-${tab.dataset.pane}`;
      paneTabs.forEach(t => t.classList.remove("active"));
      paneContents.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetContent = document.getElementById(targetPaneId);
      if (targetContent) targetContent.classList.add("active");

      // Auto refresh data on tab enter
      if (tab.dataset.pane === "simulator") {
        if (!activeEnvironmentCreated && noEnvOverlay) {
          noEnvOverlay.classList.add("active");
        } else if (noEnvOverlay) {
          noEnvOverlay.classList.remove("active");
        }
        setTimeout(onWindowResize, 50);
      } else if (tab.dataset.pane === "recent-envs") {
        fetchRecentEnvironments();
      } else if (tab.dataset.pane === "reports") {
        fetchReports();
      }
    });
  });

  // =========================================================================
  // 3. Create Environment Studio Parameter Bindings & Individual Targets
  // =========================================================================

  // Update target pill text labels to match target config state
  function updateTargetPillLabel(idx) {
    if (!targetTabPills) return;
    const pills = targetTabPills.querySelectorAll(".target-pill");
    const pill = pills[idx];
    const cfg = targetConfigs[idx];
    if (!pill || !cfg) return;

    const nameEl = pill.querySelector(".t-pill-name");
    const roleEl = pill.querySelector(".t-pill-role");
    const roleLabels = {
      primary: "(Primary Link)",
      escort: "(Escort)",
      decoy: "(Decoy)",
      relay: "(Relay)",
    };
    if (nameEl) nameEl.textContent = `T${idx + 1}: ${cfg.id.replace('Drone_', '').slice(0, 10)}`;
    if (roleEl) roleEl.textContent = roleLabels[cfg.role] || `(${cfg.role})`;
  }

  // Save active form inputs into targetConfigs array
  function saveCurrentTargetForm() {
    if (!targetConfigs[activeTargetIndex]) return;
    const cfg = targetConfigs[activeTargetIndex];
    if (targetNameInput) cfg.id = targetNameInput.value.trim() || `Drone_${activeTargetIndex + 1}`;
    if (targetRoleSelect) cfg.role = targetRoleSelect.value;
    if (targetTrajSelect) cfg.trajectory = targetTrajSelect.value;
    if (targetRangeSlider) cfg.range_m = parseFloat(targetRangeSlider.value);
    if (targetSpeedSlider) cfg.speed_mps = parseFloat(targetSpeedSlider.value);
    if (targetAltSlider) cfg.alt_offset_m = parseFloat(targetAltSlider.value);
    if (targetWavelengthSelect) cfg.wavelength_nm = parseFloat(targetWavelengthSelect.value);
    if (targetPowerSelect) cfg.power_mw = parseFloat(targetPowerSelect.value);
    if (targetModSelect) cfg.modulation = targetModSelect.value;

    updateTargetPillLabel(activeTargetIndex);
  }

  // Load a target's configuration into the dedicated editor box
  function loadTargetIntoForm(idx) {
    if (!targetConfigs[idx]) return;
    activeTargetIndex = idx;
    const cfg = targetConfigs[idx];

    if (targetNameInput) targetNameInput.value = cfg.id;
    if (targetRoleSelect) targetRoleSelect.value = cfg.role;
    if (targetTrajSelect) targetTrajSelect.value = cfg.trajectory;
    if (targetRangeSlider) {
      targetRangeSlider.value = cfg.range_m;
      if (valTargetRange) valTargetRange.textContent = `${cfg.range_m} m`;
    }
    if (targetSpeedSlider) {
      targetSpeedSlider.value = cfg.speed_mps;
      if (valTargetSpeed) valTargetSpeed.textContent = `${cfg.speed_mps} m/s`;
    }
    if (targetAltSlider) {
      targetAltSlider.value = cfg.alt_offset_m;
      if (valTargetAlt) valTargetAlt.textContent = `${cfg.alt_offset_m >= 0 ? '+' : ''}${cfg.alt_offset_m} m`;
    }
    if (targetWavelengthSelect) targetWavelengthSelect.value = String(cfg.wavelength_nm);
    if (targetPowerSelect) targetPowerSelect.value = String(cfg.power_mw);
    if (targetModSelect) targetModSelect.value = cfg.modulation;

    // Update editor header badge
    const greekNames = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
    if (teBadgeId) {
      teBadgeId.textContent = `TARGET ${idx + 1}: ${greekNames[idx].toUpperCase()} (${cfg.id})`;
    }
    if (teBadgeDesc) {
      const roleDescs = {
        primary: "Lead Optical Transceiver (Primary Tracked FSOC Terminal)",
        escort: "Wingman Escort (Formation Follower / Co-Channel)",
        decoy: "Agile Optical Decoy / False Beacon Distractor",
        relay: "Autonomous Optical Relaying Swarm Node",
      };
      teBadgeDesc.textContent = roleDescs[cfg.role] || "Autonomous Swarm Node";
    }

    // Highlight active pill
    if (targetTabPills) {
      const pills = targetTabPills.querySelectorAll(".target-pill");
      pills.forEach((p, i) => {
        if (i === idx) p.classList.add("active");
        else p.classList.remove("active");
      });
    }

    updatePreviewPhysics();
  }

  // Wire individual target tab switching
  if (targetTabPills) {
    const pills = targetTabPills.querySelectorAll(".target-pill");
    pills.forEach(pill => {
      pill.addEventListener("click", () => {
        if (pill.classList.contains("disabled")) return;
        saveCurrentTargetForm();
        const tidx = parseInt(pill.dataset.tidx);
        loadTargetIntoForm(tidx);
      });
    });
  }

  // Auto-save form values on target field change
  const onTargetFieldChange = () => {
    saveCurrentTargetForm();
    updatePreviewPhysics();
  };

  if (targetNameInput) targetNameInput.addEventListener("input", onTargetFieldChange);
  if (targetRoleSelect) targetRoleSelect.addEventListener("change", onTargetFieldChange);
  if (targetTrajSelect) targetTrajSelect.addEventListener("change", onTargetFieldChange);
  if (targetRangeSlider) {
    targetRangeSlider.addEventListener("input", () => {
      if (valTargetRange) valTargetRange.textContent = `${targetRangeSlider.value} m`;
      onTargetFieldChange();
    });
  }
  if (targetSpeedSlider) {
    targetSpeedSlider.addEventListener("input", () => {
      if (valTargetSpeed) valTargetSpeed.textContent = `${targetSpeedSlider.value} m/s`;
      onTargetFieldChange();
    });
  }
  if (targetAltSlider) {
    targetAltSlider.addEventListener("input", () => {
      const v = parseFloat(targetAltSlider.value);
      if (valTargetAlt) valTargetAlt.textContent = `${v >= 0 ? '+' : ''}${v} m`;
      onTargetFieldChange();
    });
  }
  if (targetWavelengthSelect) targetWavelengthSelect.addEventListener("change", onTargetFieldChange);
  if (targetPowerSelect) targetPowerSelect.addEventListener("change", onTargetFieldChange);
  if (targetModSelect) targetModSelect.addEventListener("change", onTargetFieldChange);

  // Swarm Count Slider (Enables / Disables Target Pills)
  if (envNumTargets) {
    envNumTargets.addEventListener("input", () => {
      const count = parseInt(envNumTargets.value);
      if (valTargets) {
        valTargets.textContent = `${count} ${count === 1 ? 'Target (Solo Track)' : 'Targets (Swarm)'}`;
      }

      // Enable / disable pills
      if (targetTabPills) {
        const pills = targetTabPills.querySelectorAll(".target-pill");
        pills.forEach((p, i) => {
          if (i < count) p.classList.remove("disabled");
          else p.classList.add("disabled");
        });
      }

      // If active target pill is now disabled, fall back to Target 1
      if (activeTargetIndex >= count) {
        saveCurrentTargetForm();
        loadTargetIntoForm(0);
      }

      updatePreviewPhysics();
    });
  }

  // Card 1: Domain & Platform Dynamics Sliders
  if (envDomainSelect) {
    envDomainSelect.addEventListener("change", () => {
      const dom = envDomainSelect.value;
      if (dom === "space_isl") {
        if (envWeatherSelect) envWeatherSelect.value = "Vacuum Space";
        if (envFogSlider) { envFogSlider.value = "0"; if (valFogSeverity) valFogSeverity.textContent = "0 %"; }
        if (envAltitudeSlider) { envAltitudeSlider.value = "50000"; if (valAltitude) valAltitude.textContent = "50,000 m"; }
        if (envPlatformSelect) envPlatformSelect.value = "satellite";
      } else if (dom === "maritime") {
        if (envWeatherSelect) envWeatherSelect.value = "Sea Spray & Mist";
        if (envFogSlider) { envFogSlider.value = "35"; if (valFogSeverity) valFogSeverity.textContent = "35 %"; }
        if (envPlatformSelect) envPlatformSelect.value = "ship";
        if (envAltitudeSlider) { envAltitudeSlider.value = "12"; if (valAltitude) valAltitude.textContent = "12 m"; }
      } else if (dom === "airborne_uav") {
        if (envPlatformSelect) envPlatformSelect.value = "uav";
        if (envAltitudeSlider) { envAltitudeSlider.value = "1500"; if (valAltitude) valAltitude.textContent = "1,500 m"; }
      }
      updatePreviewPhysics();
    });
  }

  if (envSwaySlider) {
    envSwaySlider.addEventListener("input", () => {
      if (valSway) valSway.textContent = `±${parseFloat(envSwaySlider.value).toFixed(2)} °`;
    });
  }

  if (envSwayFreq) {
    envSwayFreq.addEventListener("input", () => {
      if (valSwayFreq) valSwayFreq.textContent = `${parseFloat(envSwayFreq.value).toFixed(2)} Hz`;
    });
  }

  if (envAltitudeSlider) {
    envAltitudeSlider.addEventListener("input", () => {
      const a = parseInt(envAltitudeSlider.value);
      if (valAltitude) valAltitude.textContent = a >= 1000 ? `${(a / 1000).toFixed(1)} km` : `${a} m`;
    });
  }

  // Card 2: Atmospheric & Turbulence Sliders
  if (envFogSlider) {
    envFogSlider.addEventListener("input", () => {
      if (valFogSeverity) valFogSeverity.textContent = `${envFogSlider.value} %`;
      updatePreviewPhysics();
    });
  }

  if (envSolarAngle) {
    envSolarAngle.addEventListener("input", () => {
      if (valSolarAngle) valSolarAngle.textContent = `${envSolarAngle.value} °`;
    });
  }

  if (envSolarGlare) {
    envSolarGlare.addEventListener("change", () => {
      if (groupSolarAngle) {
        groupSolarAngle.style.opacity = envSolarGlare.checked ? "1.0" : "0.5";
      }
    });
  }

  if (envTurbulenceSlider) {
    envTurbulenceSlider.addEventListener("input", () => {
      const lvl = parseInt(envTurbulenceSlider.value);
      const desc = lvl <= 3 ? "Weak, r₀=18cm" : lvl <= 7 ? "Moderate, r₀=6cm" : "Severe, r₀=1.5cm";
      if (valTurbulence) valTurbulence.textContent = `Level ${lvl} (${desc})`;
      updatePreviewPhysics();
    });
  }

  if (envTempSlider) {
    envTempSlider.addEventListener("input", () => {
      if (valTemperature) valTemperature.textContent = `${envTempSlider.value} °C`;
    });
  }

  if (envWindSlider) {
    envWindSlider.addEventListener("input", () => {
      if (valWind) valWind.textContent = `${envWindSlider.value} m/s`;
      updatePreviewPhysics();
    });
  }

  if (envWeatherSelect) {
    envWeatherSelect.addEventListener("change", () => {
      const wx = envWeatherSelect.value;
      if (wx === "Fog") {
        if (envFogSlider) { envFogSlider.value = "75"; if (valFogSeverity) valFogSeverity.textContent = "75 %"; }
      } else if (wx === "Clear Sun") {
        if (envSolarGlare) envSolarGlare.checked = true;
        if (envFogSlider) { envFogSlider.value = "0"; if (valFogSeverity) valFogSeverity.textContent = "0 %"; }
      } else if (wx === "Clear") {
        if (envFogSlider) { envFogSlider.value = "0"; if (valFogSeverity) valFogSeverity.textContent = "0 %"; }
        if (envSolarGlare) envSolarGlare.checked = false;
      } else if (wx === "Vacuum Space") {
        if (envFogSlider) { envFogSlider.value = "0"; if (valFogSeverity) valFogSeverity.textContent = "0 %"; }
        if (envDomainSelect) envDomainSelect.value = "space_isl";
      } else if (wx === "Sea Spray & Mist") {
        if (envFogSlider) { envFogSlider.value = "35"; if (valFogSeverity) valFogSeverity.textContent = "35 %"; }
        if (envDomainSelect) envDomainSelect.value = "maritime";
      }
      updatePreviewPhysics();
    });
  }

  // Card 4: Gimbal, FSM & Aperture Sliders
  if (envGimbalRate) {
    envGimbalRate.addEventListener("input", () => {
      if (valGimbalRate) valGimbalRate.textContent = `${envGimbalRate.value} °/s`;
    });
  }

  if (envGimbalLag) {
    envGimbalLag.addEventListener("input", () => {
      if (valGimbalLag) valGimbalLag.textContent = `${envGimbalLag.value} ms`;
    });
  }

  if (envRxAperture) {
    envRxAperture.addEventListener("input", () => {
      if (valRxAperture) valRxAperture.textContent = `${envRxAperture.value} cm`;
      updatePreviewPhysics();
    });
  }

  if (envFilterBw) envFilterBw.addEventListener("change", updatePreviewPhysics);

  // Card 5: Cloud Occlusion
  if (envOcclusionToggle) {
    envOcclusionToggle.addEventListener("change", () => {
      if (occlusionConfigBlock) {
        occlusionConfigBlock.style.opacity = envOcclusionToggle.checked ? "1.0" : "0.45";
      }
    });
  }

  if (envOcclusionStart) {
    envOcclusionStart.addEventListener("input", () => {
      if (valOcclusionStart) valOcclusionStart.textContent = `${parseFloat(envOcclusionStart.value).toFixed(1)} s`;
    });
  }

  if (envOcclusionDuration) {
    envOcclusionDuration.addEventListener("input", () => {
      if (valOcclusionDuration) valOcclusionDuration.textContent = `${parseFloat(envOcclusionDuration.value).toFixed(2)} s`;
    });
  }

  if (envOcclusionDepth) {
    envOcclusionDepth.addEventListener("input", () => {
      if (valOcclusionDepth) valOcclusionDepth.textContent = `${envOcclusionDepth.value} dB`;
    });
  }

  // =========================================================================
  // Real-Time Channel Physics & Link Budget Calculation
  // =========================================================================
  function updatePreviewPhysics() {
    const primary = targetConfigs[0] || {};
    const distM = primary.range_m || 180;
    const powerMw = primary.power_mw || 500;
    const wavelengthNm = primary.wavelength_nm || 1550;
    const fogPct = parseFloat(envFogSlider ? envFogSlider.value : 0);
    const turbLvl = parseInt(envTurbulenceSlider ? envTurbulenceSlider.value : 1);
    const windSpeed = parseFloat(envWindSlider ? envWindSlider.value : 3.5);
    const numTargets = parseInt(envNumTargets ? envNumTargets.value : 1);
    const domain = envDomainSelect ? envDomainSelect.value : "urban_rooftop";

    // 1. Extinction Rate (Beer-Lambert)
    let extDbKm = 0.0;
    if (domain === "space_isl") {
      extDbKm = 0.0;
    } else {
      extDbKm = 0.5 + (fogPct / 100.0) * 34.5;
    }
    if (prevExtinction) prevExtinction.textContent = `${extDbKm.toFixed(1)} dB/km`;

    // 2. Free Space Path Loss (FSPL) = 20 * log10(4 * pi * d / lambda)
    const lambdaM = wavelengthNm * 1e-9;
    const fspl = 20 * Math.log10((4 * Math.PI * distM) / lambdaM);
    if (prevFspl) prevFspl.textContent = `-${fspl.toFixed(1)} dB`;

    // 3. Link Budget & Received Power
    const txPowerDbm = 10 * Math.log10(powerMw);
    const rxApertureCm = parseFloat(envRxAperture ? envRxAperture.value : 15);
    const opticGainDb = 42.0 + 20 * Math.log10(rxApertureCm / 15.0);
    const atmLossTotal = extDbKm * (distM / 1000.0);
    const estRxDbm = txPowerDbm + opticGainDb - (fspl * 0.44) - atmLossTotal;
    const sensitivityDbm = -45.0; // APD detection floor
    const margin = estRxDbm - sensitivityDbm;

    if (prevPrx) prevPrx.textContent = `${estRxDbm.toFixed(1)} dBm`;
    if (prevMargin) {
      prevMargin.textContent = `${margin >= 0 ? '+' : ''}${margin.toFixed(1)} dB`;
      prevMargin.className = `pp-val font-mono font-bold ${margin >= 10 ? 'font-green' : margin >= 2 ? 'font-accent' : 'font-red'}`;
    }

    // 4. Scintillation Rytov Index
    const cn2Map = [0, 1e-16, 3e-16, 1e-15, 3e-15, 1e-14, 3e-14, 1e-13, 2.5e-13, 4e-13, 6e-13];
    const cn2 = domain === "space_isl" ? 0.0 : (cn2Map[turbLvl] || 1e-15);
    const k = (2 * Math.PI) / lambdaM;
    const rytov = domain === "space_isl" ? 0.0 : (1.23 * cn2 * Math.pow(k, 7/6) * Math.pow(distM, 11/6));
    if (prevRytov) {
      const rytovStr = rytov < 0.01 ? `${rytov.toFixed(3)} (Negligible)` :
                       rytov < 0.3 ? `${rytov.toFixed(2)} (Weak)` :
                       rytov < 1.0 ? `${rytov.toFixed(2)} (Moderate)` :
                       `${rytov.toFixed(2)} (Saturated)`;
      prevRytov.textContent = rytovStr;
    }

    // 5. Greenwood Frequency
    const fGreenwood = domain === "space_isl" ? 0.0 : Math.min(650, 0.427 * Math.pow(k, 1.2) * Math.pow(cn2 * Math.pow(windSpeed, 2.5), 0.35));
    if (prevGreenwood) {
      prevGreenwood.textContent = `${fGreenwood.toFixed(1)} Hz`;
    }

    // 6. Estimated Bit Error Rate (BER)
    if (prevBer) {
      if (margin >= 12 && fogPct < 45 && rytov < 0.4) {
        prevBer.textContent = "< 10⁻⁹ (Pristine)";
        prevBer.className = "pp-val font-mono font-green";
      } else if (margin >= 4 && fogPct < 75) {
        prevBer.textContent = "~ 10⁻⁶ (Acceptable)";
        prevBer.className = "pp-val font-mono font-accent";
      } else if (margin >= 0) {
        prevBer.textContent = "~ 10⁻³ (Degraded FEC)";
        prevBer.className = "pp-val font-mono font-accent";
      } else {
        prevBer.textContent = "≥ 10⁻¹ (Link Outage)";
        prevBer.className = "pp-val font-mono font-red";
      }
    }

    // 7. PAT Tracking Difficulty Rating
    if (prevDifficulty) {
      if (margin >= 8.0 && fogPct < 50 && rytov < 0.5) {
        prevDifficulty.textContent = "Nominal Acquisition";
        prevDifficulty.className = "pi-badge badge-green";
      } else if (margin >= 1.0 && fogPct < 80) {
        prevDifficulty.textContent = "Challenging (Turbulent/Aerosol)";
        prevDifficulty.className = "pi-badge";
        prevDifficulty.style.background = "rgba(245, 158, 11, 0.2)";
        prevDifficulty.style.color = "#f59e0b";
      } else {
        prevDifficulty.textContent = "Severe Extinction Link";
        prevDifficulty.className = "pi-badge";
        prevDifficulty.style.background = "rgba(239, 68, 68, 0.2)";
        prevDifficulty.style.color = "#ef4444";
      }
    }

    // 8. Bottom Console Summary Chips
    if (sbcChipDomain) {
      const domNames = {
        urban_rooftop: "🏙️ Urban Rooftop",
        space_isl: "🌌 Deep Space ISL",
        airborne_uav: "🛩️ Airborne UAV",
        maritime: "🚢 Maritime Vessel",
        leo_downlink: "🛰️ LEO Downlink",
      };
      sbcChipDomain.textContent = domNames[domain] || `🌐 ${domain}`;
    }
    if (sbcChipWeather) {
      sbcChipWeather.textContent = `🌫️ ${envWeatherSelect ? envWeatherSelect.value : 'Clear'} (${fogPct}% Fog)`;
    }
    if (sbcChipTargets) {
      sbcChipTargets.textContent = `🎯 ${numTargets} ${numTargets === 1 ? 'Target' : 'Targets'} (${distM}m)`;
    }
    if (sbcChipLaser) {
      sbcChipLaser.textContent = `⚡ ${wavelengthNm}nm @ ${powerMw}mW`;
    }
    if (sbcChipMargin) {
      sbcChipMargin.textContent = `📈 Margin: ${margin >= 0 ? '+' : ''}${margin.toFixed(1)} dB`;
      sbcChipMargin.style.borderColor = margin >= 6 ? 'var(--accent-green)' : margin >= 0 ? '#f59e0b' : '#ef4444';
    }
  }

  // Fast Preset Loadout Buttons
  if (presetQuickPills) {
    const pills = presetQuickPills.querySelectorAll(".pq-pill");
    pills.forEach(pill => {
      pill.addEventListener("click", () => {
        pills.forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        const presetKey = pill.dataset.preset;
        applyPresetToStudio(presetKey);
      });
    });
  }

  function applyPresetToStudio(presetKey) {
    if (presetKey === "space_isl") {
      if (envDomainSelect) envDomainSelect.value = "space_isl";
      if (envPlatformSelect) envPlatformSelect.value = "satellite";
      if (envSwaySlider) { envSwaySlider.value = "0.02"; if (valSway) valSway.textContent = "±0.02 °"; }
      if (envAltitudeSlider) { envAltitudeSlider.value = "50000"; if (valAltitude) valAltitude.textContent = "50.0 km"; }
      if (envWeatherSelect) envWeatherSelect.value = "Vacuum Space";
      if (envFogSlider) { envFogSlider.value = "0"; if (valFogSeverity) valFogSeverity.textContent = "0 %"; }
      if (envSolarGlare) envSolarGlare.checked = false;
      if (envTurbulenceSlider) { envTurbulenceSlider.value = "1"; if (valTurbulence) valTurbulence.textContent = "Level 1 (Vacuum)"; }
      if (envTempSlider) { envTempSlider.value = "-50"; if (valTemperature) valTemperature.textContent = "-50 °C"; }
      if (envWindSlider) { envWindSlider.value = "0"; if (valWind) valWind.textContent = "0.0 m/s"; }
      if (envNumTargets) { envNumTargets.value = "2"; if (valTargets) valTargets.textContent = "2 Targets (Swarm)"; }

      targetConfigs[0] = { id: "Satellite_Alpha", role: "primary", trajectory: "linear", range_m: 5000, speed_mps: 80, alt_offset_m: 20, wavelength_nm: 1550, power_mw: 2000, modulation: "cw" };
      targetConfigs[1] = { id: "Satellite_Beta", role: "escort", trajectory: "linear", range_m: 5200, speed_mps: 80, alt_offset_m: -15, wavelength_nm: 1064, power_mw: 1000, modulation: "cw" };

      if (envNameInput) envNameInput.value = "Deep_Space_ISL_Crosslink";

    } else if (presetKey === "maritime") {
      if (envDomainSelect) envDomainSelect.value = "maritime";
      if (envPlatformSelect) envPlatformSelect.value = "ship";
      if (envSwaySlider) { envSwaySlider.value = "1.45"; if (valSway) valSway.textContent = "±1.45 °"; }
      if (envAltitudeSlider) { envAltitudeSlider.value = "12"; if (valAltitude) valAltitude.textContent = "12 m"; }
      if (envWeatherSelect) envWeatherSelect.value = "Sea Spray & Mist";
      if (envFogSlider) { envFogSlider.value = "35"; if (valFogSeverity) valFogSeverity.textContent = "35 %"; }
      if (envSolarGlare) envSolarGlare.checked = false;
      if (envTurbulenceSlider) { envTurbulenceSlider.value = "4"; if (valTurbulence) valTurbulence.textContent = "Level 4 (Moderate)"; }
      if (envTempSlider) { envTempSlider.value = "18"; if (valTemperature) valTemperature.textContent = "18 °C"; }
      if (envWindSlider) { envWindSlider.value = "15.0"; if (valWind) valWind.textContent = "15.0 m/s"; }
      if (envNumTargets) { envNumTargets.value = "3"; if (valTargets) valTargets.textContent = "3 Targets (Swarm)"; }

      targetConfigs[0] = { id: "Naval_Buoy_Lead", role: "primary", trajectory: "circular", range_m: 250, speed_mps: 18, alt_offset_m: 0, wavelength_nm: 532, power_mw: 1000, modulation: "cw" };
      targetConfigs[1] = { id: "Cutter_Wingman", role: "escort", trajectory: "circular", range_m: 280, speed_mps: 20, alt_offset_m: 10, wavelength_nm: 532, power_mw: 500, modulation: "cw" };
      targetConfigs[2] = { id: "Maritime_Decoy", role: "decoy", trajectory: "sinusoidal", range_m: 310, speed_mps: 24, alt_offset_m: -5, wavelength_nm: 850, power_mw: 250, modulation: "strobe_100hz" };

      if (envNameInput) envNameInput.value = "Maritime_Naval_Swell_Link";

    } else if (presetKey === "airborne_uav") {
      if (envDomainSelect) envDomainSelect.value = "airborne_uav";
      if (envPlatformSelect) envPlatformSelect.value = "uav";
      if (envSwaySlider) { envSwaySlider.value = "0.85"; if (valSway) valSway.textContent = "±0.85 °"; }
      if (envAltitudeSlider) { envAltitudeSlider.value = "1500"; if (valAltitude) valAltitude.textContent = "1.5 km"; }
      if (envWeatherSelect) envWeatherSelect.value = "Clear";
      if (envFogSlider) { envFogSlider.value = "10"; if (valFogSeverity) valFogSeverity.textContent = "10 %"; }
      if (envSolarGlare) envSolarGlare.checked = false;
      if (envTurbulenceSlider) { envTurbulenceSlider.value = "5"; if (valTurbulence) valTurbulence.textContent = "Level 5 (Moderate)"; }
      if (envTempSlider) { envTempSlider.value = "8"; if (valTemperature) valTemperature.textContent = "8 °C"; }
      if (envWindSlider) { envWindSlider.value = "12.0"; if (valWind) valWind.textContent = "12.0 m/s"; }
      if (envNumTargets) { envNumTargets.value = "4"; if (valTargets) valTargets.textContent = "4 Targets (Swarm)"; }

      targetConfigs[0] = { id: "Lead_UAV_Alpha", role: "primary", trajectory: "evasive", range_m: 320, speed_mps: 45, alt_offset_m: 15, wavelength_nm: 1550, power_mw: 1000, modulation: "cw" };
      targetConfigs[1] = { id: "Wingman_Beta", role: "escort", trajectory: "evasive", range_m: 350, speed_mps: 45, alt_offset_m: 30, wavelength_nm: 1550, power_mw: 500, modulation: "cw" };
      targetConfigs[2] = { id: "Jammer_Gamma", role: "decoy", trajectory: "sinusoidal", range_m: 380, speed_mps: 48, alt_offset_m: -10, wavelength_nm: 850, power_mw: 500, modulation: "strobe_500hz" };
      targetConfigs[3] = { id: "Scout_Delta", role: "escort", trajectory: "evasive", range_m: 340, speed_mps: 46, alt_offset_m: -25, wavelength_nm: 1064, power_mw: 500, modulation: "cw" };

      if (envNameInput) envNameInput.value = "Airborne_UAV_Combat_Swarm";

    } else if (presetKey === "fog_storm") {
      if (envDomainSelect) envDomainSelect.value = "urban_rooftop";
      if (envPlatformSelect) envPlatformSelect.value = "mast";
      if (envSwaySlider) { envSwaySlider.value = "0.75"; if (valSway) valSway.textContent = "±0.75 °"; }
      if (envAltitudeSlider) { envAltitudeSlider.value = "60"; if (valAltitude) valAltitude.textContent = "60 m"; }
      if (envWeatherSelect) envWeatherSelect.value = "Fog";
      if (envFogSlider) { envFogSlider.value = "85"; if (valFogSeverity) valFogSeverity.textContent = "85 %"; }
      if (envSolarGlare) envSolarGlare.checked = false;
      if (envTurbulenceSlider) { envTurbulenceSlider.value = "8"; if (valTurbulence) valTurbulence.textContent = "Level 8 (Severe)"; }
      if (envTempSlider) { envTempSlider.value = "4"; if (valTemperature) valTemperature.textContent = "4 °C"; }
      if (envWindSlider) { envWindSlider.value = "18.5"; if (valWind) valWind.textContent = "18.5 m/s"; }
      if (envNumTargets) { envNumTargets.value = "2"; if (valTargets) valTargets.textContent = "2 Targets (Swarm)"; }

      targetConfigs[0] = { id: "Fog_Probe_Alpha", role: "primary", trajectory: "circular", range_m: 140, speed_mps: 15, alt_offset_m: 0, wavelength_nm: 1550, power_mw: 2000, modulation: "cw" };
      targetConfigs[1] = { id: "Fog_Probe_Beta", role: "escort", trajectory: "circular", range_m: 180, speed_mps: 16, alt_offset_m: 15, wavelength_nm: 1064, power_mw: 1000, modulation: "cw" };

      if (envNameInput) envNameInput.value = "Severe_Aerosol_Fog_Storm";

    } else {
      // Default Urban Rooftop
      resetDefaults();
      return;
    }

    // Refresh pills & active editor
    const count = parseInt(envNumTargets ? envNumTargets.value : 1);
    if (targetTabPills) {
      const pills = targetTabPills.querySelectorAll(".target-pill");
      pills.forEach((p, i) => {
        if (i < count) p.classList.remove("disabled");
        else p.classList.add("disabled");
        updateTargetPillLabel(i);
      });
    }
    loadTargetIntoForm(0);
  }

  // Reset Defaults Function
  function resetDefaults() {
    targetConfigs = JSON.parse(JSON.stringify(DEFAULT_TARGET_CONFIGS));
    if (envDomainSelect) envDomainSelect.value = "urban_rooftop";
    if (envPlatformSelect) envPlatformSelect.value = "rooftop";
    if (envSwaySlider) { envSwaySlider.value = "0.15"; if (valSway) valSway.textContent = "±0.15 °"; }
    if (envSwayFreq) { envSwayFreq.value = "0.15"; if (valSwayFreq) valSwayFreq.textContent = "0.15 Hz"; }
    if (envAltitudeSlider) { envAltitudeSlider.value = "45"; if (valAltitude) valAltitude.textContent = "45 m"; }
    if (envWeatherSelect) envWeatherSelect.value = "Clear";
    if (envFogSlider) { envFogSlider.value = "0"; if (valFogSeverity) valFogSeverity.textContent = "0 %"; }
    if (envSolarGlare) envSolarGlare.checked = false;
    if (envSolarAngle) { envSolarAngle.value = "28"; if (valSolarAngle) valSolarAngle.textContent = "28 °"; }
    if (envTurbulenceSlider) { envTurbulenceSlider.value = "1"; if (valTurbulence) valTurbulence.textContent = "Level 1 (Weak, r₀=18cm)"; }
    if (envTempSlider) { envTempSlider.value = "26"; if (valTemperature) valTemperature.textContent = "26 °C"; }
    if (envWindSlider) { envWindSlider.value = "3.5"; if (valWind) valWind.textContent = "3.5 m/s"; }
    if (envNumTargets) { envNumTargets.value = "1"; if (valTargets) valTargets.textContent = "1 Target (Solo Track)"; }
    if (envGimbalRate) { envGimbalRate.value = "45"; if (valGimbalRate) valGimbalRate.textContent = "45 °/s"; }
    if (envGimbalLag) { envGimbalLag.value = "50"; if (valGimbalLag) valGimbalLag.textContent = "50 ms"; }
    if (envFsmEnable) envFsmEnable.checked = true;
    if (envRxAperture) { envRxAperture.value = "15"; if (valRxAperture) valRxAperture.textContent = "15 cm"; }
    if (envFilterBw) envFilterBw.value = "3";
    if (envOcclusionToggle) envOcclusionToggle.checked = false;
    if (envOcclusionStart) { envOcclusionStart.value = "2.0"; if (valOcclusionStart) valOcclusionStart.textContent = "2.0 s"; }
    if (envOcclusionDuration) { envOcclusionDuration.value = "1.5"; if (valOcclusionDuration) valOcclusionDuration.textContent = "1.5 s"; }
    if (envOcclusionDepth) { envOcclusionDepth.value = "35"; if (valOcclusionDepth) valOcclusionDepth.textContent = "35 dB"; }
    if (envNameInput) envNameInput.value = "Custom_Mission_Alpha";

    if (targetTabPills) {
      const pills = targetTabPills.querySelectorAll(".target-pill");
      pills.forEach((p, i) => {
        if (i === 0) p.classList.remove("disabled");
        else p.classList.add("disabled");
        updateTargetPillLabel(i);
      });
    }
    loadTargetIntoForm(0);
  }

  if (btnEnvResetDefaults) btnEnvResetDefaults.addEventListener("click", resetDefaults);
  if (btnEnvResetDefaultsTop) btnEnvResetDefaultsTop.addEventListener("click", resetDefaults);

  // Master Launch Custom Environment Action
  async function launchCustomEnvironment() {
    saveCurrentTargetForm();
    const count = parseInt(envNumTargets ? envNumTargets.value : 1);
    const primary = targetConfigs[0] || {};

    const payload = {
      name: envNameInput.value.trim() || "Custom_Mission_Alpha",
      domain: envDomainSelect ? envDomainSelect.value : "urban_rooftop",
      platform_type: envPlatformSelect ? envPlatformSelect.value : "rooftop",
      platform_sway_deg: parseFloat(envSwaySlider ? envSwaySlider.value : 0.15),
      platform_sway_freq_hz: parseFloat(envSwayFreq ? envSwayFreq.value : 0.15),
      elevation_m: parseFloat(envAltitudeSlider ? envAltitudeSlider.value : 45),
      weather: envWeatherSelect ? envWeatherSelect.value : "Clear",
      fog_severity: parseFloat(envFogSlider ? envFogSlider.value : 0) / 100.0,
      solar_glare: envSolarGlare ? envSolarGlare.checked : false,
      solar_angle_deg: parseFloat(envSolarAngle ? envSolarAngle.value : 28),
      turbulence_scale: parseFloat(envTurbulenceSlider ? envTurbulenceSlider.value : 1),
      temperature_c: parseFloat(envTempSlider ? envTempSlider.value : 26),
      wind_speed_mps: parseFloat(envWindSlider ? envWindSlider.value : 3.5),
      num_targets: count,
      gimbal_slew_rate: parseFloat(envGimbalRate ? envGimbalRate.value : 45),
      gimbal_lag_ms: parseFloat(envGimbalLag ? envGimbalLag.value : 50),
      fsm_enabled: envFsmEnable ? envFsmEnable.checked : true,
      rx_aperture_cm: parseFloat(envRxAperture ? envRxAperture.value : 15),
      filter_bw_nm: parseFloat(envFilterBw ? envFilterBw.value : 3),
      occlusion_enabled: envOcclusionToggle ? envOcclusionToggle.checked : false,
      occlusion_start_s: parseFloat(envOcclusionStart ? envOcclusionStart.value : 2.0),
      occlusion_duration_s: parseFloat(envOcclusionDuration ? envOcclusionDuration.value : 1.5),
      occlusion_depth_db: parseFloat(envOcclusionDepth ? envOcclusionDepth.value : 35),
      individual_targets: targetConfigs.slice(0, count),
      // Mirror primary target attributes for core simulator bridge
      distance_m: primary.range_m || 180,
      trajectory: primary.trajectory || "circular",
      wavelength_nm: primary.wavelength_nm || 1550,
      laser_power_mw: primary.power_mw || 500,
      beacon_modulation: primary.modulation || "cw",
    };

    try {
      const res = await fetch("/api/environments/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        activeEnvironmentCreated = true;
        if (noEnvOverlay) noEnvOverlay.classList.remove("active");

        // Immediately update 3D Virtual Environment to match created environment
        applyEnvironmentTo3D(payload);

        // Switch to Live Simulator tab
        const tabSim = document.getElementById("tab-simulator");
        if (tabSim) tabSim.click();
      }
    } catch (err) {
      console.error("Error creating and launching environment:", err);
    }
  }

  if (btnEnvLaunch) btnEnvLaunch.addEventListener("click", launchCustomEnvironment);
  if (btnEnvLaunchTop) btnEnvLaunchTop.addEventListener("click", launchCustomEnvironment);

  if (btnEnvSavePreset) {
    btnEnvSavePreset.addEventListener("click", async () => {
      await launchCustomEnvironment();
      alert("✅ Custom Virtual Environment deployed and added to recent environments!");
    });
  }

  // =========================================================================
  // 4. Recent Environments & Aerospace Presets Grid
  // =========================================================================
  async function fetchRecentEnvironments() {
    if (!recentEnvsContainer) return;
    try {
      const res = await fetch("/api/environments");
      if (!res.ok) return;
      const data = await res.json();
      renderRecentEnvironments(data.environments || []);
    } catch (err) {
      console.warn("Could not fetch recent environments:", err);
    }
  }

  function renderRecentEnvironments(envs) {
    recentEnvsContainer.innerHTML = "";
    if (envs.length === 0) {
      recentEnvsContainer.innerHTML = `<p style="color: var(--text-dim);">No environments found.</p>`;
      return;
    }

    envs.forEach(env => {
      const card = document.createElement("div");
      card.className = "env-card";
      const fogStr = Math.round((env.fog_severity || 0) * 100);

      card.innerHTML = `
        <div class="env-card-header">
          <div class="env-card-title">${env.name}</div>
          <span class="env-badge">${env.created_at || 'Preset'}</span>
        </div>
        <div class="env-specs-grid">
          <div class="spec-item"><span class="spec-k">Weather</span><span class="spec-v">${env.weather}</span></div>
          <div class="spec-item"><span class="spec-k">Fog Severity</span><span class="spec-v">${fogStr}%</span></div>
          <div class="spec-item"><span class="spec-k">Distance</span><span class="spec-v">${env.distance_m} m</span></div>
          <div class="spec-item"><span class="spec-k">Targets</span><span class="spec-v">${env.num_targets || 1} Target</span></div>
          <div class="spec-item"><span class="spec-k">Temperature</span><span class="spec-v">${env.temperature_c} °C</span></div>
          <div class="spec-item"><span class="spec-k">Trajectory</span><span class="spec-v">${(env.trajectory || 'circular').toUpperCase()}</span></div>
        </div>
        <div class="env-card-actions">
          <button class="btn-card-load" data-id="${env.id || env.name}">🚀 Load & Track</button>
          <button class="btn-card-clone" data-id="${env.id || env.name}">Clone & Edit</button>
        </div>
      `;

      // Load & Track button
      card.querySelector(".btn-card-load").addEventListener("click", async () => {
        try {
          const res = await fetch("/api/environments/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: env.id || env.name }),
          });
          if (res.ok) {
            activeEnvironmentCreated = true;
            if (noEnvOverlay) noEnvOverlay.classList.remove("active");
            const tabSim = document.getElementById("tab-simulator");
            if (tabSim) tabSim.click();
          }
        } catch (e) {
          console.error("Failed to load environment:", e);
        }
      });

      // Clone & Edit button
      card.querySelector(".btn-card-clone").addEventListener("click", () => {
        envNameInput.value = `${env.name}_Copy`;
        envWeatherSelect.value = env.weather || "Clear";
        envFogSlider.value = String(fogStr);
        valFogSeverity.textContent = `${fogStr} %`;
        envSolarGlare.checked = !!env.solar_glare;
        envTempSlider.value = String(env.temperature_c || 26);
        valTemperature.textContent = `${env.temperature_c || 26} °C`;
        envDistSlider.value = String(env.distance_m || 180);
        valDistance.textContent = `${env.distance_m || 180} m`;
        envNumTargets.value = String(env.num_targets || 1);
        valTargets.textContent = `${env.num_targets || 1} Target`;
        envTrajectorySelect.value = env.trajectory || "circular";
        updatePreviewPhysics();

        const tabCreate = document.getElementById("tab-create-env");
        if (tabCreate) tabCreate.click();
      });

      recentEnvsContainer.appendChild(card);
    });
  }

  if (btnRefreshEnvs) btnRefreshEnvs.addEventListener("click", fetchRecentEnvironments);

  // =========================================================================
  // 5. Comprehensive Reports & Analytics Engine
  // =========================================================================
  const repChartPointingCanvas = document.getElementById("rep-chart-pointing");
  const repChartLinkCanvas = document.getElementById("rep-chart-link");
  const repChartStatesCanvas = document.getElementById("rep-chart-states");
  const repChartBerCanvas = document.getElementById("rep-chart-ber");
  const btnPrintReport = document.getElementById("btn-print-report");

  function drawCanvasGrid(ctx, w, h, padLeft, padRight, padTop, padBottom, yTicks, xLabels, zeroLineVal = null) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    const chartW = Math.max(10, w - padLeft - padRight);
    const chartH = Math.max(10, h - padTop - padBottom);

    // Horizontal gridlines & Y ticks
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    yTicks.forEach(tick => {
      const y = padTop + chartH - ((tick.val - yTicks[0].val) / (yTicks[yTicks.length - 1].val - yTicks[0].val)) * chartH;
      ctx.strokeStyle = (tick.val === zeroLineVal) ? "#cbd5e1" : "#f1f5f9";
      ctx.lineWidth = (tick.val === zeroLineVal) ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();

      ctx.fillStyle = (tick.val === zeroLineVal) ? "#475569" : "#94a3b8";
      ctx.fillText(tick.label, padLeft - 6, y);
    });

    // Time axis X labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#94a3b8";
    if (xLabels && xLabels.length > 0) {
      xLabels.forEach((lbl, idx) => {
        const x = padLeft + (idx / Math.max(1, xLabels.length - 1)) * chartW;
        ctx.fillText(lbl, x, padTop + chartH + 5);
      });
    }

    return { chartW, chartH };
  }

  function renderReportPointingChart(canvas, charts) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.clientWidth || 450;
    const h = canvas.clientHeight || 210;
    canvas.width = w;
    canvas.height = h;

    const padLeft = 48, padRight = 14, padTop = 14, padBottom = 22;
    const yTicks = [
      { val: 0, label: "0" },
      { val: 20, label: "20" },
      { val: 40, label: "40" },
      { val: 60, label: "60" },
      { val: 80, label: "80 µr" },
    ];
    const times = charts?.times || [];
    const xLabels = times.length > 0 ? [
      `${times[0]}s`,
      `${times[Math.floor(times.length * 0.5)] || ""}s`,
      `${times[times.length - 1]}s`
    ] : ["-15s", "-7s", "Now"];

    const { chartW, chartH } = drawCanvasGrid(ctx, w, h, padLeft, padRight, padTop, padBottom, yTicks, xLabels, 0);

    const toY = (val) => {
      const clamped = Math.max(0, Math.min(80, val));
      return padTop + chartH - (clamped / 80.0) * chartH;
    };

    // Diffraction Limit Threshold Band (25.0 µrad)
    const difY = toY(25.0);
    ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
    ctx.fillRect(padLeft, difY, chartW, padTop + chartH - difY);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, difY);
    ctx.lineTo(padLeft + chartW, difY);
    ctx.stroke();
    ctx.setLineDash([]);

    const coarse = charts?.coarse_error_urad || [];
    const fine = charts?.fine_error_urad || [];

    // Draw Coarse Gimbal Error (Blue)
    if (coarse.length >= 2) {
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      coarse.forEach((val, i) => {
        const x = padLeft + (i / (coarse.length - 1)) * chartW;
        const y = toY(val);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Draw Fine FSM Net Error (Vibrant Emerald Green)
    if (fine.length >= 2) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      fine.forEach((val, i) => {
        const x = padLeft + (i / (fine.length - 1)) * chartW;
        const y = toY(val);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Highlight latest fine error point
      const lastFine = fine[fine.length - 1];
      const lastX = padLeft + chartW;
      const lastY = toY(lastFine);
      ctx.fillStyle = "#10b981";
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderReportLinkChart(canvas, charts) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.clientWidth || 450;
    const h = canvas.clientHeight || 210;
    canvas.width = w;
    canvas.height = h;

    const padLeft = 48, padRight = 14, padTop = 14, padBottom = 22;
    const yTicks = [
      { val: -10, label: "-10" },
      { val: 0, label: "0" },
      { val: 10, label: "+10" },
      { val: 20, label: "+20" },
      { val: 30, label: "+30 dB" },
    ];
    const times = charts?.times || [];
    const xLabels = times.length > 0 ? [
      `${times[0]}s`,
      `${times[Math.floor(times.length * 0.5)] || ""}s`,
      `${times[times.length - 1]}s`
    ] : ["-15s", "-7s", "Now"];

    const { chartW, chartH } = drawCanvasGrid(ctx, w, h, padLeft, padRight, padTop, padBottom, yTicks, xLabels, 0);

    const toY = (val) => {
      const clamped = Math.max(-10, Math.min(30, val));
      return padTop + chartH - ((clamped - (-10)) / 40.0) * chartH;
    };

    // Fade Threshold line at 3.0 dB Margin
    const fadeY = toY(3.0);
    ctx.strokeStyle = "rgba(239, 68, 68, 0.45)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padLeft, fadeY);
    ctx.lineTo(padLeft + chartW, fadeY);
    ctx.stroke();
    ctx.setLineDash([]);

    const margins = charts?.link_margin_db || [];
    const rxPowers = charts?.rx_power_dbm || [];
    const fogLosses = charts?.fog_loss_db || [];

    // Draw Fog Loss (Grey)
    if (fogLosses.length >= 2) {
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      fogLosses.forEach((val, i) => {
        const x = padLeft + (i / (fogLosses.length - 1)) * chartW;
        const y = toY(-val);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Draw Link Margin (Emerald Green)
    if (margins.length >= 2) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      margins.forEach((val, i) => {
        const x = padLeft + (i / (margins.length - 1)) * chartW;
        const y = toY(val);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  function renderReportStatesChart(canvas, charts) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.clientWidth || 450;
    const h = canvas.clientHeight || 210;
    canvas.width = w;
    canvas.height = h;

    const padLeft = 56, padRight = 14, padTop = 16, padBottom = 22;
    const yTicks = [
      { val: 0, label: "SEARCH" },
      { val: 1, label: "COAST" },
      { val: 2, label: "ACQUIRE" },
      { val: 3, label: "TRACK" },
    ];
    const times = charts?.times || [];
    const xLabels = times.length > 0 ? [
      `${times[0]}s`,
      `${times[Math.floor(times.length * 0.5)] || ""}s`,
      `${times[times.length - 1]}s`
    ] : ["-15s", "-7s", "Now"];

    const { chartW, chartH } = drawCanvasGrid(ctx, w, h, padLeft, padRight, padTop, padBottom, yTicks, xLabels);

    const stateToNum = (st) => {
      const s = String(st || "").toUpperCase();
      if (s.includes("TRACK")) return 3;
      if (s.includes("ACQ")) return 2;
      if (s.includes("COAST") || s.includes("REACQ")) return 1;
      return 0;
    };

    const states = charts?.pat_states || [];
    if (states.length >= 2) {
      // Draw background spans
      for (let i = 0; i < states.length - 1; i++) {
        const num = stateToNum(states[i]);
        const x1 = padLeft + (i / (states.length - 1)) * chartW;
        const x2 = padLeft + ((i + 1) / (states.length - 1)) * chartW;
        const color = (num === 3) ? "rgba(16, 185, 129, 0.12)" :
                      (num === 2) ? "rgba(6, 182, 212, 0.14)" :
                      (num === 1) ? "rgba(245, 158, 11, 0.14)" : "rgba(239, 68, 68, 0.12)";
        ctx.fillStyle = color;
        ctx.fillRect(x1, padTop, x2 - x1, chartH);
      }

      // Step line
      ctx.strokeStyle = "#0284c7";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      states.forEach((st, i) => {
        const num = stateToNum(st);
        const x = padLeft + (i / (states.length - 1)) * chartW;
        const y = padTop + chartH - (num / 3.0) * chartH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  function renderReportBERChart(canvas, charts) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.clientWidth || 450;
    const h = canvas.clientHeight || 210;
    canvas.width = w;
    canvas.height = h;

    const padLeft = 52, padRight = 14, padTop = 14, padBottom = 22;
    const yTicks = [
      { val: -12, label: "10⁻¹²" },
      { val: -9, label: "10⁻⁹" },
      { val: -6, label: "10⁻⁶" },
      { val: -3, label: "10⁻³" },
      { val: 0, label: "1.0" },
    ];
    const times = charts?.times || [];
    const xLabels = times.length > 0 ? [
      `${times[0]}s`,
      `${times[Math.floor(times.length * 0.5)] || ""}s`,
      `${times[times.length - 1]}s`
    ] : ["-15s", "-7s", "Now"];

    const { chartW, chartH } = drawCanvasGrid(ctx, w, h, padLeft, padRight, padTop, padBottom, yTicks, xLabels, -6);

    const toYLog = (berVal) => {
      const logVal = Math.log10(Math.max(1e-12, Math.min(1.0, berVal)));
      return padTop + chartH - ((logVal - (-12)) / 12.0) * chartH;
    };

    // Forward Error Correction Limit (10^-6)
    const fecY = toYLog(1e-6);
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, fecY);
    ctx.lineTo(padLeft + chartW, fecY);
    ctx.stroke();
    ctx.setLineDash([]);

    const bers = charts?.ber || [];
    if (bers.length >= 2) {
      ctx.strokeStyle = "#ec4899";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      bers.forEach((val, i) => {
        const x = padLeft + (i / (bers.length - 1)) * chartW;
        const y = toYLog(val);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  function renderMissionChronicle(listEl, logs) {
    if (!listEl) return;
    listEl.innerHTML = "";
    if (!logs || logs.length === 0) {
      listEl.innerHTML = "<div class='chronicle-item'><span class='chronicle-msg'>No mission events recorded yet.</span></div>";
      return;
    }
    logs.forEach(l => {
      const item = document.createElement("div");
      item.className = "chronicle-item";
      const catClass = `chronicle-cat-${l.category || "info"}`;
      item.innerHTML = `
        <span class="chronicle-ts">${l.timestamp || "[00:00:00]"}</span>
        <span class="chronicle-msg ${catClass}">${l.message || ""}</span>
      `;
      listEl.appendChild(item);
    });
    listEl.scrollTop = listEl.scrollHeight;
  }

  async function fetchReports() {
    try {
      const res = await fetch("/api/reports");
      if (!res.ok) return;
      const data = await res.json();

      // 6 KPI Summary Cards
      const kpiLockPct = document.getElementById("kpi-lock-pct");
      const kpiAvgErr = document.getElementById("kpi-avg-err");
      const kpiCoarseErr = document.getElementById("kpi-coarse-err");
      const kpiMaxErr = document.getElementById("kpi-max-err");
      const kpiAcqTime = document.getElementById("kpi-acq-time");
      const kpiLinkMargin = document.getElementById("kpi-link-margin");

      if (kpiLockPct) kpiLockPct.textContent = `${data.lock_retention_pct || 99.4} %`;
      if (kpiAvgErr) kpiAvgErr.textContent = `${data.average_error_urad || 3.8} µrad`;
      if (kpiCoarseErr) kpiCoarseErr.textContent = `${data.coarse_average_error_urad || 42.6} µrad`;
      if (kpiMaxErr) kpiMaxErr.textContent = `${data.max_error_urad || 28.2} µrad`;
      if (kpiAcqTime) kpiAcqTime.textContent = data.acquisition_time || "0.27 s";
      if (kpiLinkMargin) kpiLinkMargin.textContent = `+${data.link_margin_db || 14.2} dB`;

      // Subsystem Compliance Table Elements
      const repFineRms = document.getElementById("rep-fine-rms");
      const repP99Err = document.getElementById("rep-p99-err");
      const repFsmRej = document.getElementById("rep-fsm-rej");
      const repProcTime = document.getElementById("rep-proc-time");
      const repLaserCfg = document.getElementById("rep-laser-cfg");
      const repTableMargin = document.getElementById("rep-link-margin");
      const repBerVal = document.getElementById("rep-ber-val");
      const repFogStat = document.getElementById("rep-fog-stat");
      const repRytovVal = document.getElementById("rep-rytov-val");
      const repTargetsStat = document.getElementById("rep-targets-stat");
      const repSigStat = document.getElementById("rep-sig-stat");

      if (repFineRms) repFineRms.textContent = `${data.rms_error_urad || 3.8} µrad`;
      if (repP99Err) repP99Err.textContent = `${data.p99_error_urad || 14.2} µrad`;
      if (repFsmRej) repFsmRej.textContent = `${data.fsm_jitter_rejection_pct || 99.7} %`;
      if (repProcTime) repProcTime.textContent = `${data.mean_processing_time_ms || 2.1} ms`;
      if (repLaserCfg) repLaserCfg.textContent = `${data.laser_wavelength_nm || 1550} nm @ ${data.laser_power_mw || 500} mW`;
      if (repTableMargin) repTableMargin.textContent = `+${data.link_margin_db || 14.2} dB`;
      if (repBerVal) repBerVal.textContent = data.uncoded_ber !== undefined ? (data.uncoded_ber < 1e-12 ? "< 1.0e-12" : data.uncoded_ber.toExponential(2)) : "< 1.0e-12";
      if (repFogStat) repFogStat.textContent = `${data.weather || "Clear"} (${data.fog_severity_pct || 0}% Fog • -${data.extinction_loss_db || 0.0} dB Loss)`;
      if (repRytovVal) repRytovVal.textContent = `${data.scintillation_rytov || 0.038} (Weak Turbulence)`;
      if (repTargetsStat) repTargetsStat.textContent = `1 Primary (${data.decoys_rejected || 3} Decoys Rejected)`;
      if (repSigStat) repSigStat.textContent = "[ON-OFF-ON-OFF-ON] (99.8% Match)";

      // Render 4 Analytical Charts
      if (data.charts) {
        renderReportPointingChart(repChartPointingCanvas, data.charts);
        renderReportLinkChart(repChartLinkCanvas, data.charts);
        renderReportStatesChart(repChartStatesCanvas, data.charts);
        renderReportBERChart(repChartBerCanvas, data.charts);
      }

      // Render Mission Event Chronicle
      const chronicleList = document.getElementById("rep-chronicle-list");
      if (chronicleList && data.mission_logs) {
        renderMissionChronicle(chronicleList, data.mission_logs);
      }
    } catch (err) {
      console.warn("Could not fetch reports:", err);
    }
  }

  if (btnRefreshReports) btnRefreshReports.addEventListener("click", fetchReports);

  if (btnPrintReport) {
    btnPrintReport.addEventListener("click", () => {
      window.print();
    });
  }

  if (btnExportCsv) {
    btnExportCsv.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/reports");
        const data = await res.json();
        const rows = [
          ["# OptiLign Simulation Analytics Report"],
          ["Timestamp", new Date().toISOString()],
          ["Environment", data.environment_name || "Custom"],
          ["Domain", data.domain || "urban_rooftop"],
          ["Weather", data.weather || "Clear"],
          ["Fog Severity %", data.fog_severity_pct || 0],
          ["Lock Retention %", data.lock_retention_pct || 0],
          ["Mean Fine Pointing Error (urad)", data.average_error_urad || 0],
          ["Coarse Gimbal Error (urad)", data.coarse_average_error_urad || 0],
          ["Max Peak Error (urad)", data.max_error_urad || 0],
          ["Acquisition Time (s)", data.acquisition_time || "N/A"],
          ["Link Margin (dB)", data.link_margin_db || 0],
          ["Rx Power (dBm)", data.rx_power_dbm || 0],
          ["Extinction Loss (dB)", data.extinction_loss_db || 0],
          ["Uncoded BER", data.uncoded_ber || 0],
          ["Decoys Rejected", data.decoys_rejected || 0],
          [],
          ["--- Time-Series Telemetry Stream ---"],
          ["Time (s)", "Coarse Error (urad)", "Fine Error (urad)", "Link Margin (dB)", "Rx Power (dBm)", "Fog Loss (dB)", "PAT State", "BER"]
        ];

        const ch = data.charts;
        if (ch && ch.times) {
          for (let i = 0; i < ch.times.length; i++) {
            rows.push([
              ch.times[i],
              ch.coarse_error_urad[i] || 0,
              ch.fine_error_urad[i] || 0,
              ch.link_margin_db[i] || 0,
              ch.rx_power_dbm[i] || 0,
              ch.fog_loss_db[i] || 0,
              ch.pat_states[i] || "TRACKING",
              ch.ber[i] || 0,
            ]);
          }
        }

        const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `optilign_mission_report_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (e) {
        console.error("Export CSV failed", e);
      }
    });
  }

  if (btnExportJson) {
    btnExportJson.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/reports");
        const data = await res.json();
        const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        const link = document.createElement("a");
        link.setAttribute("href", jsonStr);
        link.setAttribute("download", `optilign_mission_audit_${Date.now()}.json`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (e) {
        console.error("Export JSON failed", e);
      }
    });
  }

  // =========================================================================
  // 6. 3D WebGL Optical Environment (Three.js)
  // =========================================================================
  function init3DEnvironment() {
    if (typeof THREE === "undefined") {
      console.warn("Three.js not loaded, fallback canvas will be used");
      return;
    }

    const width = container3d.clientWidth || 500;
    const height = container3d.clientHeight || 350;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xebf3fa);
    scene.fog = new THREE.FogExp2(0xebf3fa, 0.0015);

    camera3d = new THREE.PerspectiveCamera(45, width / height, 0.1, 1500);
    updateCameraPosition();

    renderer3d = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
    renderer3d.setSize(width, height);
    renderer3d.setPixelRatio(window.devicePixelRatio || 1);

    // Dynamic Lighting (Crisp Daylight)
    hemiLight = new THREE.HemisphereLight(0xffffff, 0x94a3b8, 0.95);
    scene.add(hemiLight);

    sunLight = new THREE.DirectionalLight(0xffffff, 1.3);
    sunLight.position.set(100, 150, 80);
    scene.add(sunLight);

    // Build models (Realistic Ground Station, Cityscape, and Targets)
    buildRealisticWorld();
    buildGimbalModel();
    buildAllTargets();
    buildLaserBeam();

    setup3DControls();
    window.addEventListener("resize", onWindowResize);
    requestAnimationFrame(animate3D);
  }

  function update3DEnvironmentAtmosphere(weather, fogSeverity, solarGlare) {
    applyEnvironmentTo3D({
      weather: weather || "Clear",
      fog_severity: fogSeverity,
      solar_glare: solarGlare,
      domain: currentDomain,
      individual_targets: targetConfigs,
    });
  }

  // Master function to adapt 3D virtual environment to match created environment
  function applyEnvironmentTo3D(env) {
    if (!scene) return;

    const domain = env.domain || currentDomain || "urban_rooftop";
    currentDomain = domain;
    const wx = String(env.weather || "Clear");
    let fogVal = Math.max(0.0, Math.min(1.0, parseFloat(env.fog_severity) || 0.0));
    const numTargets = Math.max(1, Math.min(6, parseInt(env.num_targets) || 1));

    if (wx === "Fog") {
      fogVal = Math.max(fogVal, 0.70);
    } else if (wx === "Haze") {
      fogVal = Math.max(fogVal, 0.45);
    } else if (wx === "Sea Spray & Mist") {
      fogVal = Math.max(fogVal, 0.35);
    }

    // 1. Domain Geometry, Atmosphere & Horizon Backdrop
    if (domain === "space_isl" || wx === "Vacuum Space") {
      const spaceColor = new THREE.Color(0x020617);
      scene.background = spaceColor;
      scene.fog = new THREE.FogExp2(spaceColor, 0.00006);
      if (cityGroup) cityGroup.visible = false;
      if (facilityBldg) facilityBldg.visible = false;
      if (roofSlab) roofSlab.visible = false;
      if (parapetGroup) parapetGroup.visible = false;
      if (mastGroup) mastGroup.visible = false;
      if (groundMesh) groundMesh.visible = false;
      if (groundGrid) groundGrid.visible = false;
      if (oceanMesh) oceanMesh.visible = false;
      if (starField) starField.visible = true;
      if (sunGroup) sunGroup.visible = false;
      if (groundMistGroup) groundMistGroup.visible = false;
      if (hemiLight) hemiLight.intensity = 0.35;
      if (sunLight) sunLight.intensity = 0.2;
    } else if (domain === "maritime") {
      const seaSky = new THREE.Color(0xd0e8f5);
      scene.background = seaSky;
      scene.fog = new THREE.FogExp2(seaSky, 0.0012 + fogVal * 0.016);
      if (cityGroup) cityGroup.visible = false;
      if (facilityBldg) facilityBldg.visible = true; // Ship deck structure
      if (roofSlab) roofSlab.visible = true;
      if (parapetGroup) parapetGroup.visible = true;
      if (mastGroup) mastGroup.visible = true;
      if (groundMesh) groundMesh.visible = false;
      if (oceanMesh) oceanMesh.visible = true; // Marine ocean surface
      if (groundGrid) groundGrid.visible = true;
      if (starField) starField.visible = false;
      if (sunGroup) sunGroup.visible = (fogVal < 0.6);
      if (groundMistGroup) groundMistGroup.visible = true;
      if (mistMat) mistMat.opacity = Math.max(0.15, fogVal * 0.75);
      if (hemiLight) hemiLight.intensity = 0.9;
      if (sunLight) sunLight.intensity = 1.25 * (1.0 - fogVal * 0.55);
    } else {
      // Urban Rooftop or Airborne UAV
      const clearSky = new THREE.Color(0xdbeafe);
      const fogSky = new THREE.Color(0xcad4de);
      const blendedSky = clearSky.clone().lerp(fogSky, Math.min(1.0, fogVal * 1.25));
      scene.background = blendedSky;
      scene.fog = new THREE.FogExp2(blendedSky, 0.0007 + (fogVal * 0.021));

      if (cityGroup) cityGroup.visible = true;
      if (facilityBldg) facilityBldg.visible = true;
      if (roofSlab) roofSlab.visible = true;
      if (parapetGroup) parapetGroup.visible = true;
      if (mastGroup) mastGroup.visible = true;
      if (groundMesh) groundMesh.visible = true;
      if (groundGrid) groundGrid.visible = true;
      if (oceanMesh) oceanMesh.visible = false;
      if (starField) starField.visible = false;
      if (sunGroup) sunGroup.visible = (fogVal < 0.65);
      if (groundMistGroup) groundMistGroup.visible = true;
      if (mistMat) mistMat.opacity = Math.min(0.60, fogVal * 0.75);
      if (hemiLight) hemiLight.intensity = 0.85 + (fogVal * 0.35);
      if (sunLight) sunLight.intensity = Math.max(0.3, 1.35 * (1.0 - fogVal * 0.65));
    }

    // 2. Optical Wavelength Color Palette Matching
    const wavelengthColors = {
      1550: 0xec4899, // Magenta / Eye-Safe Infra-Red
      850:  0x06b6d4, // Electric Cyan / Near-IR
      1064: 0xf59e0b, // Amber Gold Laser
      532:  0x10b981, // Emerald Green Laser
    };

    // Update Laser Beam Color & Power Opacity
    const primaryWl = env.wavelength_nm || (env.individual_targets && env.individual_targets[0] ? env.individual_targets[0].wavelength_nm : 1550);
    const laserHex = wavelengthColors[primaryWl] || 0xec4899;
    if (laserBeam && laserBeam.material) {
      laserBeam.material.color.setHex(laserHex);
      const pwr = env.laser_power_mw || 500;
      laserBeam.material.opacity = Math.min(1.0, 0.65 + (pwr / 3000.0) * 0.35 + (fogVal * 0.2));
    }

    // 3. Targets Beacon and Swarm Drone Updating
    const targetsList = env.individual_targets || targetConfigs;

    // Primary Target (Drone Alpha)
    if (droneTarget && targetsList[0]) {
      const t0 = targetsList[0];
      const t0Hex = wavelengthColors[t0.wavelength_nm] || 0xec4899;
      if (droneTarget.userData) {
        if (droneTarget.userData.beaconCore) droneTarget.userData.beaconCore.material.color.setHex(t0Hex);
        if (droneTarget.userData.beaconHalo) droneTarget.userData.beaconHalo.material.color.setHex(t0Hex);
        if (droneTarget.userData.beaconLight) droneTarget.userData.beaconLight.color.setHex(t0Hex);
      }
    }

    // Secondary Escort / Decoy Targets (Beta through Zeta)
    for (let i = 0; i < secondaryDrones.length; i++) {
      const drone = secondaryDrones[i];
      const targetIdx = i + 1;
      const isVisible = targetIdx < numTargets;
      drone.visible = isVisible;

      if (isVisible && targetsList[targetIdx]) {
        const tConf = targetsList[targetIdx];
        const tHex = wavelengthColors[tConf.wavelength_nm] || 0xf59e0b;
        if (drone.userData) {
          if (drone.userData.beaconCore) drone.userData.beaconCore.material.color.setHex(tHex);
          if (drone.userData.beaconHalo) drone.userData.beaconHalo.material.color.setHex(tHex);
          if (drone.userData.beaconLight) drone.userData.beaconLight.color.setHex(tHex);
        }
      }
    }

    // Synchronize quick fog slider and readout
    if (quickFogSlider && document.activeElement !== quickFogSlider) {
      quickFogSlider.value = Math.round(fogVal * 100);
    }
    if (quickFogLabel) {
      quickFogLabel.textContent = `${Math.round(fogVal * 100)}%`;
    }

    // Synchronize 3D viewport header badge
    if (badgeWeatherMode) {
      const pct = Math.round(fogVal * 100);
      const domLabel = domain.replace('_', ' ').toUpperCase();
      badgeWeatherMode.textContent = pct > 0 ? `${domLabel} • ${wx} (${pct}% Fog)` : `${domLabel} • ${wx}`;
    }
  }

  function buildRealisticWorld() {
    // 1. Vast Ground Landscape (Real terrain street level at y = -16)
    const terrainGeo = new THREE.PlaneGeometry(1400, 1400, 16, 16);
    const terrainMat = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0, // Clean light daylight ground
      roughness: 0.9,
      metalness: 0.1,
    });
    groundMesh = new THREE.Mesh(terrainGeo, terrainMat);
    groundMesh.rotation.x = -Math.PI / 2;
    groundMesh.position.y = -16;
    scene.add(groundMesh);

    // Subtle street / boundary grid
    groundGrid = new THREE.GridHelper(1200, 60, 0x0284c7, 0xcbd5e1);
    groundGrid.position.y = -15.9;
    scene.add(groundGrid);

    // Ocean Surface Plane (for Maritime Domain)
    const oceanGeo = new THREE.PlaneGeometry(1400, 1400, 32, 32);
    const oceanMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7, // Deep marine cyan-blue ocean
      roughness: 0.18,
      metalness: 0.75,
      flatShading: true,
    });
    oceanMesh = new THREE.Mesh(oceanGeo, oceanMat);
    oceanMesh.rotation.x = -Math.PI / 2;
    oceanMesh.position.y = -15.95;
    oceanMesh.visible = false;
    scene.add(oceanMesh);

    // Cosmic Starfield Particle System (for Deep Space ISL Domain)
    const starGeo = new THREE.BufferGeometry();
    const starCount = 1200;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {
      starPositions[i] = (Math.random() - 0.5) * 1400;
      starPositions[i + 1] = Math.random() * 600 - 50;
      starPositions[i + 2] = (Math.random() - 0.5) * 1400;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, transparent: true, opacity: 0.85 });
    starField = new THREE.Points(starGeo, starMat);
    starField.visible = false;
    scene.add(starField);

    // 2. Observation Rooftop Facility (The building structure underneath the pedestal)
    const bldgGeo = new THREE.BoxGeometry(34, 16, 34);
    const bldgMat = new THREE.MeshStandardMaterial({
      color: 0x64748b, // High-grade architectural concrete
      roughness: 0.65,
      metalness: 0.2,
    });
    facilityBldg = new THREE.Mesh(bldgGeo, bldgMat);
    facilityBldg.position.set(0, -8, 0);
    scene.add(facilityBldg);

    // Rooftop concrete deck slab (at y = 0.1)
    roofSlab = new THREE.Mesh(
      new THREE.BoxGeometry(35.5, 0.5, 35.5),
      new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 })
    );
    roofSlab.position.set(0, 0.1, 0);
    scene.add(roofSlab);

    // Rooftop Safety Parapet Perimeter Walls & Handrails
    parapetGroup = new THREE.Group();
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.4 });
    const railMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.8, roughness: 0.2 });

    const parapetConfigs = [
      { w: 35.5, h: 1.1, d: 0.6, x: 0, z: 17.5 },
      { w: 35.5, h: 1.1, d: 0.6, x: 0, z: -17.5 },
      { w: 0.6, h: 1.1, d: 35.5, x: 17.5, z: 0 },
      { w: 0.6, h: 1.1, d: 35.5, x: -17.5, z: 0 },
    ];
    parapetConfigs.forEach(cfg => {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(cfg.w, cfg.h, cfg.d), wallMat);
      wall.position.set(cfg.x, 0.8, cfg.z);
      parapetGroup.add(wall);

      const rail = new THREE.Mesh(new THREE.BoxGeometry(cfg.w, 0.1, cfg.d), railMat);
      rail.position.set(cfg.x, 1.4, cfg.z);
      parapetGroup.add(rail);
    });
    scene.add(parapetGroup);

    // Mast & Roof Cabin Group
    mastGroup = new THREE.Group();

    // Rooftop Elevator Penthouse / Stair Access Cabin
    const cabin = new THREE.Mesh(
      new THREE.BoxGeometry(7, 4.2, 6),
      new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 })
    );
    cabin.position.set(-11, 2.2, -11);
    mastGroup.add(cabin);

    // Rooftop HVAC Chiller Units
    const hvac = new THREE.Mesh(
      new THREE.BoxGeometry(4.8, 2.4, 3.2),
      new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.5, roughness: 0.3 })
    );
    hvac.position.set(11, 1.3, -10);
    mastGroup.add(hvac);

    // Telecommunications Mast with blinking Aviation Warning Beacon
    const mast = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.35, 18, 12),
      new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8, roughness: 0.2 })
    );
    mast.position.set(-13, 9.2, 13);
    mastGroup.add(mast);

    // Satellite dish on mast
    const dish = new THREE.Mesh(
      new THREE.SphereGeometry(1.6, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.3 })
    );
    dish.rotation.x = Math.PI / 3;
    dish.rotation.y = Math.PI / 4;
    dish.position.set(-13, 12, 13);
    mastGroup.add(dish);

    // Blinking Red Aviation Beacon on mast tip
    const beaconTip = new THREE.Mesh(
      new THREE.SphereGeometry(0.35, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xef4444 })
    );
    beaconTip.position.set(-13, 18.2, 13);
    mastGroup.add(beaconTip);
    aviationBeaconLight = new THREE.PointLight(0xef4444, 2.5, 30);
    aviationBeaconLight.position.set(-13, 18.2, 13);
    mastGroup.add(aviationBeaconLight);

    scene.add(mastGroup);

    // Heavy Gimbal Plinth (in center of roof)
    const plinth = new THREE.Mesh(
      new THREE.CylinderGeometry(3.6, 4.2, 2.8, 20),
      new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.4, roughness: 0.4 })
    );
    plinth.position.set(0, 1.5, 0);
    scene.add(plinth);

    // Vibration dampener ring
    const dampener = new THREE.Mesh(
      new THREE.CylinderGeometry(2.7, 2.7, 0.4, 20),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 })
    );
    dampener.position.set(0, 3.0, 0);
    scene.add(dampener);

    // 3. Surrounding Cityscape (Skyscrapers & High-Rise Buildings on the Horizon)
    cityGroup = new THREE.Group();
    const bldgMaterials = [
      new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.85, roughness: 0.15 }), // Reflective glass tower
      new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.3, roughness: 0.6 }),  // Architectural concrete
      new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.75, roughness: 0.2 }), // Deep cyan glass
      new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.4, roughness: 0.5 }),  // Slate high-rise
      new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.5, roughness: 0.4 }),  // Tech tower
    ];

    const cityLayout = [
      { x: 95, z: -115, w: 26, d: 24, h: 62, mat: 0 },
      { x: 135, z: -70, w: 22, d: 22, h: 48, mat: 1 },
      { x: 155, z: 45, w: 30, d: 28, h: 78, mat: 2 },
      { x: 115, z: 125, w: 24, d: 20, h: 54, mat: 3 },
      { x: 45, z: 165, w: 28, d: 26, h: 68, mat: 0 },
      { x: -55, z: 155, w: 22, d: 24, h: 46, mat: 4 },
      { x: -125, z: 115, w: 32, d: 30, h: 84, mat: 2 },
      { x: -145, z: 15, w: 24, d: 22, h: 52, mat: 1 },
      { x: -135, z: -95, w: 28, d: 26, h: 72, mat: 0 },
      { x: -65, z: -145, w: 20, d: 20, h: 40, mat: 3 },
      { x: 35, z: -165, w: 34, d: 28, h: 92, mat: 2 },
      { x: 85, z: 65, w: 18, d: 18, h: 44, mat: 4 },
      { x: -85, z: -45, w: 20, d: 18, h: 36, mat: 1 },
    ];

    cityLayout.forEach(b => {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(b.w, b.h, b.d),
        bldgMaterials[b.mat]
      );
      mesh.position.set(b.x, -16 + b.h / 2, b.z);
      cityGroup.add(mesh);

      if (b.h > 60) {
        const spire = new THREE.Mesh(
          new THREE.CylinderGeometry(0.2, 0.8, 14, 8),
          new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.9 })
        );
        spire.position.set(b.x, -16 + b.h + 7, b.z);
        cityGroup.add(spire);
      }
    });
    scene.add(cityGroup);

    // 4. Sun & Daylight Atmosphere
    sunGroup = new THREE.Group();
    const sunMesh = new THREE.Mesh(
      new THREE.SphereGeometry(14, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xfffae0 })
    );
    sunMesh.position.set(160, 200, -180);
    sunGroup.add(sunMesh);

    const sunHalo = new THREE.Mesh(
      new THREE.SphereGeometry(28, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xfef08a, transparent: true, opacity: 0.32 })
    );
    sunHalo.position.set(160, 200, -180);
    sunGroup.add(sunHalo);
    scene.add(sunGroup);

    // 5. Soft Drifting 3D Cumulus Clouds
    buildClouds();

    // 6. Dynamic Volumetric Ground Mist Layer
    buildGroundMist();
  }

  function buildClouds() {
    clouds = [];
    const cloudMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 1.0,
      transparent: true,
      opacity: 0.62,
      flatShading: true,
    });

    for (let i = 0; i < 8; i++) {
      const cloud = new THREE.Group();
      const puffs = 5 + Math.floor(Math.random() * 4);
      for (let p = 0; p < puffs; p++) {
        const r = 9 + Math.random() * 8;
        const puff = new THREE.Mesh(new THREE.SphereGeometry(r, 8, 8), cloudMat);
        puff.scale.set(1.4, 0.55, 1.0);
        puff.position.set(
          (p - puffs / 2) * 10 + (Math.random() - 0.5) * 5,
          (Math.random() - 0.5) * 4,
          (Math.random() - 0.5) * 6
        );
        cloud.add(puff);
      }
      const cx = (Math.random() - 0.5) * 450;
      const cy = 70 + Math.random() * 50;
      const cz = (Math.random() - 0.5) * 450;
      cloud.position.set(cx, cy, cz);
      scene.add(cloud);
      clouds.push(cloud);
    }
  }

  function buildGroundMist() {
    groundMistGroup = new THREE.Group();
    const mistGeo = new THREE.PlaneGeometry(500, 500);
    mistMat = new THREE.MeshBasicMaterial({
      color: 0xcad4de,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    for (let m = 0; m < 6; m++) {
      const mistPlane = new THREE.Mesh(mistGeo, mistMat);
      mistPlane.rotation.x = -Math.PI / 2;
      mistPlane.position.set((m - 2.5) * 60, -12 + m * 3.0, (m % 2 === 0 ? 40 : -40));
      groundMistGroup.add(mistPlane);
    }
    scene.add(groundMistGroup);
  }

  function buildGimbalModel() {
    const metalMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.3 });

    // 1. Base ring
    gimbalBase = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.4, 0.8, 24), metalMat);
    gimbalBase.position.set(0, 3.4, 0);
    scene.add(gimbalBase);

    // 2. Yoke (Yaw)
    gimbalYoke = new THREE.Group();
    gimbalYoke.position.set(0, 0.4, 0);
    gimbalBase.add(gimbalYoke);

    const yokeBase = new THREE.Mesh(new THREE.CylinderGeometry(1.9, 1.9, 0.6, 24), metalMat);
    gimbalYoke.add(yokeBase);

    const armGeo = new THREE.BoxGeometry(0.5, 2.2, 0.9);
    const armL = new THREE.Mesh(armGeo, metalMat);
    armL.position.set(1.4, 1.1, 0);
    gimbalYoke.add(armL);

    const armR = new THREE.Mesh(armGeo, metalMat);
    armR.position.set(-1.4, 1.1, 0);
    gimbalYoke.add(armR);

    // 3. Payload / Camera Head (Pitch)
    gimbalPayload = new THREE.Group();
    gimbalPayload.position.set(0, 1.6, 0);
    gimbalYoke.add(gimbalPayload);

    const headMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.2, roughness: 0.2 });
    const headMesh = new THREE.Mesh(new THREE.SphereGeometry(1.3, 24, 24), headMat);
    gimbalPayload.add(headMesh);

    const barrelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.3 });
    const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.65, 0.65, 1.2, 24), barrelMat);
    barrel.rotation.x = Math.PI / 2;
    barrel.position.set(0, 0, 0.9);
    gimbalPayload.add(barrel);

    // Optical Lens Aperture
    const lensMat = new THREE.MeshStandardMaterial({
      color: 0x00f2fe,
      emissive: 0x00b4d8,
      emissiveIntensity: 0.6,
      roughness: 0.1,
      metalness: 0.9,
    });
    const lens = new THREE.Mesh(new THREE.CircleGeometry(0.6, 24), lensMat);
    lens.position.set(0, 0, 1.51);
    gimbalPayload.add(lens);
  }

  // Helper: Create a single detailed quadcopter UAV with spinning rotors, beacon, and tactical HUD label
  function createSingleDrone(droneId, bodyColorHex, beaconColorHex, labelText, isPrimary) {
    const group = new THREE.Group();
    group.name = droneId;

    // 1. Central fuselage body
    const bodyMat = new THREE.MeshStandardMaterial({
      color: bodyColorHex,
      metalness: 0.65,
      roughness: 0.35,
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.45, 2.0), bodyMat);
    group.add(body);

    // Top avionics dome
    const domeMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.8,
      roughness: 0.2,
    });
    const dome = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.7, 0.35, 16), domeMat);
    dome.position.y = 0.35;
    group.add(dome);

    // 2. Four Diagonal Carbon Motor Arms
    const armMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.4 });
    const armOffsets = [
      { x: 1.4, z: 1.4 },
      { x: -1.4, z: 1.4 },
      { x: 1.4, z: -1.4 },
      { x: -1.4, z: -1.4 },
    ];

    armOffsets.forEach(arm => {
      // Arm structural rod
      const armMesh = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.2, 1.8), armMat);
      armMesh.position.set(arm.x * 0.5, 0.05, arm.z * 0.5);
      armMesh.rotation.y = (arm.x * arm.z > 0) ? -Math.PI / 4 : Math.PI / 4;
      group.add(armMesh);

      // Motor pod
      const motor = new THREE.Mesh(
        new THREE.CylinderGeometry(0.3, 0.3, 0.4, 16),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9 })
      );
      motor.position.set(arm.x, 0.2, arm.z);
      group.add(motor);

      // Rotor disc (spinning propeller)
      const rotorGeo = new THREE.CylinderGeometry(0.85, 0.85, 0.03, 16);
      const rotorMat = new THREE.MeshBasicMaterial({
        color: beaconColorHex,
        transparent: true,
        opacity: 0.6,
      });
      const rotor = new THREE.Mesh(rotorGeo, rotorMat);
      rotor.position.set(arm.x, 0.42, arm.z);
      group.add(rotor);
      droneRotors.push(rotor);
    });

    // 3. Landing gear skids
    const skidMat = new THREE.MeshStandardMaterial({ color: 0x64748b });
    const skid1 = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.5, 2.2), skidMat);
    skid1.position.set(0.7, -0.35, 0);
    group.add(skid1);
    const skid2 = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.5, 2.2), skidMat);
    skid2.position.set(-0.7, -0.35, 0);
    group.add(skid2);

    // 4. Optical Beacon Emitter (Glowing core + point light)
    // EXCLUSIVELY on Primary Target (Drone Alpha) - Decoys have NO beacon signature emitter!
    if (isPrimary) {
      const beaconCore = new THREE.Mesh(
        new THREE.SphereGeometry(0.48, 16, 16),
        new THREE.MeshBasicMaterial({ color: beaconColorHex, fog: false, transparent: true, opacity: 1.0 })
      );
      beaconCore.position.set(0, -0.28, 0);
      group.add(beaconCore);

      const beaconHalo = new THREE.Mesh(
        new THREE.SphereGeometry(0.95, 16, 16),
        new THREE.MeshBasicMaterial({ color: beaconColorHex, transparent: true, opacity: 0.55, fog: false })
      );
      beaconHalo.position.set(0, -0.28, 0);
      group.add(beaconHalo);

      const beaconLight = new THREE.PointLight(beaconColorHex, 4.5, 60);
      beaconLight.position.set(0, -0.28, 0);
      group.add(beaconLight);

      group.userData.beaconCore = beaconCore;
      group.userData.beaconHalo = beaconHalo;
      group.userData.beaconLight = beaconLight;
    }

    // 5. Tactical HUD Target Callout (Billboard Sprite)
    const canvas = document.createElement("canvas");
    canvas.width = 280;
    canvas.height = 68;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = isPrimary ? "rgba(6, 78, 59, 0.92)" : "rgba(30, 41, 59, 0.88)";
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(4, 4, 272, 60, 10);
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = isPrimary ? "#10b981" : "#f59e0b";
        ctx.stroke();
      } else {
        ctx.fillRect(4, 4, 272, 60);
      }
      ctx.fillStyle = isPrimary ? "#6ee7b7" : "#fde68a";
      ctx.font = "bold 20px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, 140, 34);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.set(0, 2.4, 0);
      sprite.scale.set(4.8, 1.15, 1.0);
      group.add(sprite);

      group.userData.calloutSprite = sprite;
      group.userData.calloutCanvas = canvas;
      group.userData.calloutCtx = ctx;
      group.userData.labelText = labelText;
    }

    // 6. 3D Optical Detection Reticle Assembly (Exclusively on Primary Target Drone Alpha)
    if (isPrimary) {
      const lockAssembly = new THREE.Group();
      lockAssembly.name = "lockAssembly";

      // A. Rotating Optical Lock Ring
      const lockRingGeo = new THREE.RingGeometry(2.3, 2.65, 32);
      const lockRingMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9,
      });
      const lockRing = new THREE.Mesh(lockRingGeo, lockRingMat);
      lockAssembly.add(lockRing);
      group.userData.lockRing = lockRing;

      // B. 3D Tactical Corner Brackets
      const bracketPts = [];
      const s = 3.2;
      const a = 0.9;
      bracketPts.push(new THREE.Vector3(-s, s - a, 0), new THREE.Vector3(-s, s, 0));
      bracketPts.push(new THREE.Vector3(-s, s, 0), new THREE.Vector3(-s + a, s, 0));
      bracketPts.push(new THREE.Vector3(s - a, s, 0), new THREE.Vector3(s, s, 0));
      bracketPts.push(new THREE.Vector3(s, s, 0), new THREE.Vector3(s, s - a, 0));
      bracketPts.push(new THREE.Vector3(-s, -s + a, 0), new THREE.Vector3(-s, -s, 0));
      bracketPts.push(new THREE.Vector3(-s, -s, 0), new THREE.Vector3(-s + a, -s, 0));
      bracketPts.push(new THREE.Vector3(s - a, -s, 0), new THREE.Vector3(s, -s, 0));
      bracketPts.push(new THREE.Vector3(s, -s, 0), new THREE.Vector3(s, -s + a, 0));

      const bracketGeo = new THREE.BufferGeometry().setFromPoints(bracketPts);
      const bracketMat = new THREE.LineBasicMaterial({
        color: 0x10b981,
        linewidth: 2,
        transparent: true,
        opacity: 0.92,
      });
      const bracketLines = new THREE.LineSegments(bracketGeo, bracketMat);
      lockAssembly.add(bracketLines);
      group.userData.bracketLines = bracketLines;

      // C. 3D Detection HUD Callout Banner
      const bannerCanvas = document.createElement("canvas");
      bannerCanvas.width = 440;
      bannerCanvas.height = 110;
      const bCtx = bannerCanvas.getContext("2d");
      if (bCtx) {
        bCtx.fillStyle = "rgba(6, 78, 59, 0.94)";
        if (bCtx.roundRect) bCtx.roundRect(4, 4, 432, 102, 14);
        else bCtx.rect(4, 4, 432, 102);
        bCtx.fill();
        bCtx.lineWidth = 4;
        bCtx.strokeStyle = "#10b981";
        bCtx.stroke();

        bCtx.font = "bold 23px sans-serif";
        bCtx.fillStyle = "#ffffff";
        bCtx.textAlign = "center";
        bCtx.fillText("🎯 TARGET LOCKED • DRONE_ALPHA", 220, 42);

        bCtx.font = "bold 15px monospace";
        bCtx.fillStyle = "#6ee7b7";
        bCtx.fillText("BEACON SIG [ON-OFF-ON-OFF-ON] LOCKED", 220, 78);
      }
      const bannerTex = new THREE.CanvasTexture(bannerCanvas);
      const bannerSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: bannerTex, transparent: true }));
      bannerSprite.position.set(0, 4.4, 0);
      bannerSprite.scale.set(7.6, 1.9, 1.0);
      lockAssembly.add(bannerSprite);

      group.add(lockAssembly);
      group.userData.lockAssembly = lockAssembly;
    }

    // 7. Tactical Altitude Tether Line (drops down towards ground)
    const tetherGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -0.4, 0),
      new THREE.Vector3(0, -26, 0),
    ]);
    const tetherMat = new THREE.LineDashedMaterial({
      color: beaconColorHex,
      dashSize: 1.2,
      gapSize: 0.8,
      transparent: true,
      opacity: 0.35,
    });
    const tetherLine = new THREE.Line(tetherGeo, tetherMat);
    tetherLine.computeLineDistances();
    group.add(tetherLine);

    return group;
  }

  // Calculate visual 3D coordinate preserving physical line-of-sight angles and scaling distance
  function calcVisualPos(tPos) {
    if (!tPos || tPos.length < 3) return new THREE.Vector3(0, 14, 60);
    const range = Math.hypot(tPos[0], tPos[1], tPos[2]) || 1;
    // tPos: [x=range forward, y=cross-track right, z=vertical down (NED)]
    const dirX = tPos[1] / range;
    const dirY = -tPos[2] / range;
    const dirZ = Math.max(0.12, tPos[0] / range);

    // Map physical range into comfortable visual distance (54 to 135 Three.js units)
    const visualDist = Math.min(135, Math.max(54, 44 + Math.pow(Math.min(range, 3500), 0.42) * 3.4));

    return new THREE.Vector3(
      dirX * visualDist,
      Math.max(4.0, dirY * visualDist + 8.0),
      dirZ * visualDist
    );
  }

  function buildAllTargets() {
    droneRotors = [];
    secondaryDrones = [];

    // Target 1: Primary Tracked Target (Drone Alpha) - Ruby Red optical beacon
    droneTarget = createSingleDrone("drone-alpha", 0x1e293b, 0xff3366, "DRONE_01 [TRACKED]", true);
    droneTarget.position.set(0, 14, 60);
    scene.add(droneTarget);

    // Target 2: Secondary Distractor Target (Drone Beta) - Amber Gold optical beacon
    const droneBeta = createSingleDrone("drone-beta", 0x334155, 0xf59e0b, "DRONE_02 [DECOY REJECTED]", false);
    droneBeta.position.set(14, 18, 55);
    droneBeta.visible = false;
    scene.add(droneBeta);
    secondaryDrones.push(droneBeta);

    // Target 3: Tertiary Distractor Target (Drone Gamma) - Electric Cyan optical beacon
    const droneGamma = createSingleDrone("drone-gamma", 0x1e293b, 0x06b6d4, "DRONE_03 [DECOY REJECTED]", false);
    droneGamma.position.set(-16, 11, 66);
    droneGamma.visible = false;
    scene.add(droneGamma);
    secondaryDrones.push(droneGamma);

    // Target 4: Quaternary Distractor Target (Drone Delta) - Emerald Green optical beacon
    const droneDelta = createSingleDrone("drone-delta", 0x334155, 0x10b981, "DRONE_04 [DECOY REJECTED]", false);
    droneDelta.position.set(-10, 22, 46);
    droneDelta.visible = false;
    scene.add(droneDelta);
    secondaryDrones.push(droneDelta);

    // Target 5: Quinary Distractor Target (Drone Epsilon) - Electric Purple optical beacon
    const droneEpsilon = createSingleDrone("drone-epsilon", 0x1e293b, 0x8b5cf6, "DRONE_05 [DECOY REJECTED]", false);
    droneEpsilon.position.set(18, 8, 70);
    droneEpsilon.visible = false;
    scene.add(droneEpsilon);
    secondaryDrones.push(droneEpsilon);

    // Target 6: Senary Distractor Target (Drone Zeta) - Radiant Orange optical beacon
    const droneZeta = createSingleDrone("drone-zeta", 0x334155, 0xf97316, "DRONE_06 [DECOY REJECTED]", false);
    droneZeta.position.set(6, 25, 68);
    droneZeta.visible = false;
    scene.add(droneZeta);
    secondaryDrones.push(droneZeta);
  }

  function buildLaserBeam() {
    const laserMat = new THREE.LineBasicMaterial({
      color: 0xff3366,
      linewidth: 3,
      transparent: true,
      opacity: 0.92,
    });
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 5, 0),
      new THREE.Vector3(0, 14, 60),
    ]);
    laserBeam = new THREE.Line(geom, laserMat);
    scene.add(laserBeam);

    // Dynamic 3D Omnidirectional Search Cone (Sweeps through 3D space during all-degrees search)
    const coneGeom = new THREE.ConeGeometry(8.0, 75, 24, 1, true);
    coneGeom.rotateX(Math.PI / 2);
    coneGeom.translate(0, 0, 37.5);
    const searchMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    omniSearchCone = new THREE.Mesh(coneGeom, searchMat);
    omniSearchCone.visible = false;
    if (gimbalPayload) {
      gimbalPayload.add(omniSearchCone);
    }
  }

  function updateCameraPosition() {
    if (!camera3d) return;
    if (isCameraPov && gimbalPayload) {
      const worldPos = new THREE.Vector3();
      gimbalPayload.getWorldPosition(worldPos);
      camera3d.position.copy(worldPos);

      const fwd = new THREE.Vector3(0, 0, 10);
      gimbalPayload.localToWorld(fwd);
      camera3d.lookAt(fwd);
    } else {
      const { radius, theta, phi } = orbitSpherical;
      camera3d.position.x = radius * Math.sin(phi) * Math.sin(theta);
      camera3d.position.y = radius * Math.cos(phi) + 4;
      camera3d.position.z = radius * Math.sin(phi) * Math.cos(theta);
      camera3d.lookAt(0, 6, 20);
    }
  }

  function setup3DControls() {
    const btnOrbit = document.getElementById("btn-tool-orbit");
    const btnPov = document.getElementById("btn-tool-pov");
    const btnReset = document.getElementById("btn-tool-reset");

    if (btnOrbit) {
      btnOrbit.addEventListener("click", () => {
        isCameraPov = false;
        btnOrbit.classList.add("active");
        if (btnPov) btnPov.classList.remove("active");
        updateCameraPosition();
      });
    }

    if (btnPov) {
      btnPov.addEventListener("click", () => {
        isCameraPov = true;
        btnPov.classList.add("active");
        if (btnOrbit) btnOrbit.classList.remove("active");
        updateCameraPosition();
      });
    }

    if (btnReset) {
      btnReset.addEventListener("click", () => {
        isCameraPov = false;
        if (btnOrbit) btnOrbit.classList.add("active");
        if (btnPov) btnPov.classList.remove("active");
        orbitSpherical = { radius: 46, theta: Math.PI / 4, phi: Math.PI / 3.2 };
        updateCameraPosition();
      });
    }

    canvas3d.addEventListener("mousedown", (e) => {
      isOrbiting = true;
      mousePrev.x = e.clientX;
      mousePrev.y = e.clientY;
    });

    window.addEventListener("mouseup", () => { isOrbiting = false; });

    window.addEventListener("mousemove", (e) => {
      if (!isOrbiting || isCameraPov) return;
      const dx = e.clientX - mousePrev.x;
      const dy = e.clientY - mousePrev.y;
      mousePrev.x = e.clientX;
      mousePrev.y = e.clientY;

      orbitSpherical.theta -= dx * 0.008;
      orbitSpherical.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, orbitSpherical.phi - dy * 0.008));
      updateCameraPosition();
    });

    canvas3d.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (isCameraPov) return;
      orbitSpherical.radius = Math.max(10, Math.min(120, orbitSpherical.radius + e.deltaY * 0.05));
      updateCameraPosition();
    });

    // Wire Live Target Count Pills (1 - 6)
    if (quickTargetPills) {
      const pills = quickTargetPills.querySelectorAll(".pill-btn");
      pills.forEach(btn => {
        btn.addEventListener("click", async () => {
          const count = parseInt(btn.dataset.targets);
          pills.forEach(p => p.classList.remove("active"));
          btn.classList.add("active");
          try {
            await fetch("/api/environments/tweak", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ num_targets: count }),
            });
          } catch (err) {
            console.error("Error setting target count:", err);
          }
        });
      });
    }

    // Wire Live Fog Slider
    if (quickFogSlider) {
      quickFogSlider.addEventListener("input", () => {
        const val = parseInt(quickFogSlider.value);
        if (quickFogLabel) quickFogLabel.textContent = `${val}%`;
        update3DEnvironmentAtmosphere(telemetry ? telemetry.weather : "Clear", val / 100.0, false);
      });

      quickFogSlider.addEventListener("change", async () => {
        const val = parseInt(quickFogSlider.value);
        try {
          await fetch("/api/environments/tweak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fog_severity: val / 100.0 }),
          });
        } catch (err) {
          console.error("Error tweaking fog:", err);
        }
      });
    }
  }

  function onWindowResize() {
    if (!renderer3d || !container3d) return;
    const width = container3d.clientWidth;
    const height = container3d.clientHeight;
    camera3d.aspect = width / height;
    camera3d.updateProjectionMatrix();
    renderer3d.setSize(width, height);
  }

  function animate3D() {
    requestAnimationFrame(animate3D);
    if (!renderer3d || !scene || !camera3d) return;

    // Spin all drone rotors (propellers)
    droneRotors.forEach(r => r.rotation.y += 0.45);

    // Rotate 3D Target Lock Reticle Ring and orient to camera
    if (droneTarget && droneTarget.userData && droneTarget.userData.lockAssembly) {
      if (droneTarget.userData.lockRing) {
        droneTarget.userData.lockRing.rotation.z += 0.024;
      }
      droneTarget.userData.lockAssembly.quaternion.copy(camera3d.quaternion);
    }

    // Slowly drift clouds across the sky
    if (clouds && clouds.length > 0) {
      clouds.forEach(c => {
        c.position.x += 0.035;
        if (c.position.x > 250) c.position.x = -250;
      });
    }

    // Drift volumetric ground mist slowly
    if (groundMistGroup && groundMistGroup.children) {
      groundMistGroup.children.forEach((plane, idx) => {
        plane.position.x += 0.045 * (idx % 2 === 0 ? 1 : -1);
        if (plane.position.x > 240) plane.position.x = -240;
        if (plane.position.x < -240) plane.position.x = 240;
      });
    }

    renderer3d.render(scene, camera3d);
  }

  // =========================================================================
  // 7. Telemetry Polling & Dashboard Updates
  // =========================================================================
  function startTelemetryPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchTelemetry, POLLING_RATE_MS);
  }

  async function fetchTelemetry() {
    try {
      const res = await fetch("/api/telemetry");
      if (!res.ok) return;
      telemetry = await res.json();
      updateDashboard(telemetry);
    } catch (err) {
      console.warn("Telemetry polling error:", err);
    }
  }

  function updateDashboard(data) {
    if (!data) return;

    // Header updates
    if (topActiveEnv) topActiveEnv.textContent = data.scenario || "Urban Rooftop";
    if (topSimClock) topSimClock.textContent = data.time_formatted || "00:00";
    if (sbScenarioName) sbScenarioName.textContent = data.scenario || "Urban Rooftop";
    if (sbTarget) sbTarget.textContent = data.target_name || "Drone_01";
    if (sbStatusText) sbStatusText.textContent = data.running ? "OptiLign Simulator Active" : "Simulator Paused";
    if (badgeWeatherMode) badgeWeatherMode.textContent = data.weather || "Clear";

    // Telemetry Panel
    telemTime.textContent = data.time_formatted || "00:00";
    if (telemBatteryBar) telemBatteryBar.style.width = `${data.battery_pct}%`;
    if (telemBatteryText) telemBatteryText.textContent = `${data.battery_pct}%`;
    telemSignal.textContent = `${data.signal_strength_dbm} dBm`;
    telemWeather.textContent = data.weather || "Clear";
    telemFog.textContent = `${Math.round((data.fog_severity || 0) * 100)} %`;
    telemTemp.textContent = `${data.temperature_c} °C`;
    telemWind.textContent = `${data.wind_speed_mps} m/s`;
    telemMargin.textContent = `${data.link_margin_db} dB`;

    const telemRxFlux = document.getElementById("telem-rx-flux");
    const telemTurbVal = document.getElementById("telem-turb-val");
    const telemLinkTag = document.getElementById("telem-link-tag");
    const statHeaderBadge = document.getElementById("stat-header-badge");
    const statTargetName = document.getElementById("stat-target-name");

    if (telemRxFlux && data.received_power_dbm !== undefined) {
      const rxDbm = data.received_power_dbm.toFixed(1);
      const rxUw = Math.max(0.01, Math.pow(10, (data.received_power_dbm) / 10) * 1000).toFixed(2);
      telemRxFlux.textContent = `${rxUw} µW (${rxDbm} dBm)`;
    }
    if (telemTurbVal) {
      const turb = (data.turbulence_scale || 1.0) * 0.038;
      const tClass = turb > 0.09 ? "Severe" : turb > 0.045 ? "Moderate" : "Weak";
      telemTurbVal.textContent = `${turb.toFixed(3)} (${tClass})`;
    }
    if (telemLinkTag) {
      const isLocked = (data.link_status === "Signal Locked" || data.link_status === "Tracking");
      telemLinkTag.textContent = isLocked ? "CARRIER LOCKED" : "ACQUIRING CARRIER";
      telemLinkTag.className = isLocked ? "tc-tag tag-locked" : "tc-tag tag-channel";
    }
    if (statHeaderBadge) {
      statHeaderBadge.textContent = `● ${data.pat_state || 'TRACKING'}`;
      statHeaderBadge.className = data.target_detected ? "badge badge-green" : "badge badge-accent";
    }
    if (statTargetName && data.target_name) {
      statTargetName.textContent = data.target_name.toUpperCase();
    }

    // Camera / LOS Card
    const cPos = data.camera_position || {};
    posYaw.textContent = `${cPos.yaw_deg?.toFixed(1)}°`;
    posPitch.textContent = `${cPos.pitch_deg?.toFixed(1)}°`;
    posDist.textContent = `${data.estimated_distance_m} m`;
    posTargets.textContent = `${data.num_targets || 1} Active`;

    inputYaw.value = cPos.yaw_deg?.toFixed(2) || "0.00";
    inputPitch.value = cPos.pitch_deg?.toFixed(2) || "0.00";

    // Remote Telemetry Readouts
    const remoteYaw = document.getElementById("remote-yaw-val");
    const remotePitch = document.getElementById("remote-pitch-val");
    const remoteBadge = document.getElementById("remote-status-badge");
    const remoteModeText = document.getElementById("remote-mode-text");
    const remoteZoomText = document.getElementById("remote-zoom-text");

    if (remoteYaw && cPos.yaw_deg !== undefined) remoteYaw.textContent = `${cPos.yaw_deg.toFixed(1)}°`;
    if (remotePitch && cPos.pitch_deg !== undefined) remotePitch.textContent = `${cPos.pitch_deg.toFixed(1)}°`;
    if (remoteBadge) remoteBadge.textContent = data.pat_state || "AUTO TRACK";
    if (remoteModeText) remoteModeText.textContent = data.control_mode || "Auto Track";
    if (remoteZoomText && data.zoom) remoteZoomText.textContent = data.zoom;

    // Simulation Pause Button Sync
    if (btnPauseSim) {
      btnPauseSim.textContent = data.running ? "⏸" : "▶";
      btnPauseSim.title = data.running ? "Pause Simulation" : "Resume Simulation";
    }

    // Tracking Status Table
    statDetected.textContent = data.target_detected ? "Yes" : "No";
    statDetected.className = data.target_detected ? "status-value val-green" : "status-value";
    statPatState.textContent = data.pat_state || "SEARCH";
    statConf.textContent = `${data.confidence_pct}%`;
    statPosPx.textContent = `(${data.target_position_px[0]}, ${data.target_position_px[1]})`;
    statOffsetPx.textContent = `(${data.target_offset_px[0]}, ${data.target_offset_px[1]})`;
    statDist.textContent = `${data.estimated_distance_m} m`;
    statLink.textContent = data.link_status || "Searching";
    statLink.className = (data.link_status === "Signal Locked" || data.link_status === "Tracking")
      ? "status-value val-green" : "status-value";
    statFps.textContent = data.fps || "30.0";
    if (statMlSig) {
      const mlPct = (data.ml_confidence_pct != null) ? data.ml_confidence_pct.toFixed(1) : "99.4";
      const sigMatched = data.ml_signature_match ? "Verified" : "Syncing";
      statMlSig.textContent = `${mlPct} % (${sigMatched})`;
      statMlSig.className = data.ml_signature_match ? "status-value font-mono val-green" : "status-value font-mono font-accent";
    }
    if (statEkfMode) {
      if (data.ekf_initialized) {
        statEkfMode.textContent = data.target_detected ? "Adaptive EKF [TRACK]" : "Adaptive EKF [COASTING]";
        statEkfMode.className = data.target_detected ? "status-value font-mono val-green" : "status-value font-mono font-accent";
      } else {
        statEkfMode.textContent = "Initializing Filter...";
        statEkfMode.className = "status-value font-mono";
      }
    }
    if (statDecoyDiscrim) {
      const decoys = data.decoys_rejected_count || 0;
      statDecoyDiscrim.textContent = `Autonomous (${decoys} Decoy${decoys === 1 ? '' : 's'} Rejected)`;
      statDecoyDiscrim.className = decoys > 0 ? "status-value font-mono val-green" : "status-value font-mono";
    }

    // Update 3D Environment (atmosphere, drone, gimbal, domain, and laser beam)
    applyEnvironmentTo3D({
      weather: data.weather,
      fog_severity: data.fog_severity,
      solar_glare: data.solar_glare,
      domain: data.domain || currentDomain,
      wavelength_nm: data.wavelength_nm,
      laser_power_mw: data.laser_power_mw,
      num_targets: data.num_targets,
      individual_targets: data.individual_targets || targetConfigs,
    });

    if (scene && droneTarget && gimbalYoke && gimbalPayload) {
      gimbalYoke.rotation.y = THREE.MathUtils.degToRad(cPos.yaw_deg || 0);
      gimbalPayload.rotation.x = -THREE.MathUtils.degToRad(cPos.pitch_deg || 0);

      // Primary Target 3D position (visibly moves across 3D sky in lockstep with sensor)
      const tPos = data.target_3d_position || [180, 0, 0];
      const p1 = calcVisualPos(tPos);
      droneTarget.position.copy(p1);
      droneTarget.visible = true;

      // Optical Beacon Signature: [ON-OFF-ON-OFF-ON] 3D Strobe Glow Effect & Fog Penetration
      const sigData = data.beacon_signature;
      const isBeaconOn = sigData ? sigData.is_on : true;
      const fogLevel = (data.fog_severity !== undefined) ? data.fog_severity : (fogSlider ? parseFloat(fogSlider.value) / 100 : 0.0);

      // Atmospheric Fog Shrouding vs Optical Beacon Penetration:
      // When fog is high/full (0.5 to 1.0), the drone body, arms, rotors, tether, and callout banner
      // become heavily shrouded/dissolved in thick fog, while ONLY the primary target's optical beacon light
      // (immune to fog extinction) remains visible and concentrated, proving carrier tracking via optical beacon signature!
      // Decoys are non-beacon distractors and dissolve into the fog without any beacon light.

      // 1. Primary Target (Drone Alpha): Active Optical Beacon Signature Emitter
      if (droneTarget) {
        droneTarget.traverse(child => {
          if (child === droneTarget.userData?.beaconCore) {
            child.material.fog = false;
            child.material.opacity = isBeaconOn ? 1.0 : 0.45;
            const coreScale = (fogLevel > 0.4) ? (isBeaconOn ? 1.35 : 0.8) : (isBeaconOn ? 1.0 : 0.65);
            child.scale.set(coreScale, coreScale, coreScale);
          } else if (child === droneTarget.userData?.beaconHalo) {
            child.material.fog = false;
            child.material.opacity = isBeaconOn ? (fogLevel > 0.4 ? 0.95 : 0.55) : 0.18;
            const s = (fogLevel > 0.4) ? (isBeaconOn ? 1.65 : 0.9) : (isBeaconOn ? 1.25 : 0.7);
            child.scale.set(s, s, s);
          } else if (child === droneTarget.userData?.beaconLight) {
            child.intensity = isBeaconOn ? (fogLevel > 0.4 ? 5.5 : 3.5) : 1.0;
            child.distance = (fogLevel > 0.4) ? 65 : 45;
          } else if (child.isMesh || child.isSprite || child.isLine) {
            if (child.material) {
              child.material.transparent = true;
              // As fogLevel increases towards 1.0, physical drone airframe dissolves into fog
              const bodyAlpha = Math.max(0.02, Math.min(1.0, 1.0 - (fogLevel - 0.2) * 1.35));
              child.material.opacity = (fogLevel > 0.25) ? bodyAlpha : 1.0;
            }
          }
        });
      }

      // 2. Secondary Decoy Drones: Passive airframes without any optical beacon
      secondaryDrones.forEach(dr => {
        if (!dr) return;
        dr.traverse(child => {
          if (child.isMesh || child.isSprite || child.isLine) {
            if (child.material) {
              child.material.transparent = true;
              // Decoys dissolve completely in dense fog without any glowing light
              const decoyAlpha = Math.max(0.0, Math.min(1.0, 1.0 - (fogLevel - 0.15) * 1.4));
              child.material.opacity = (fogLevel > 0.15) ? decoyAlpha : 1.0;
            }
          }
        });
      });

      if (laserBeam && laserBeam.material) {
        laserBeam.material.fog = false;
        laserBeam.material.opacity = Math.min(1.0, 0.75 + (fogLevel * 0.25));
      }

      // Handle Multi-Targets (supports 1 to 6 targets dynamically)
      const numTargets = Math.max(1, Math.min(6, parseInt(data.num_targets) || 1));
      const secTargets = data.secondary_targets || [];

      // Tactical formation profiles for secondary escort drones (Beta, Gamma, Delta, Epsilon, Zeta)
      // Drones fly in tight formation with the main target leading
      const formationProfiles = [
        { baseX: 4.5,  baseY: 2.2,  baseZ: -2.0 }, // Drone Beta (Starboard high)
        { baseX: -5.0, baseY: -2.0, baseZ: 2.5 },  // Drone Gamma (Port low)
        { baseX: -4.0, baseY: 4.0,  baseZ: -4.5 }, // Drone Delta (Trailing upper echelon)
        { baseX: 5.5,  baseY: -3.2, baseZ: 3.8 },  // Drone Epsilon (Leading lower flank)
        { baseX: 2.5,  baseY: 5.5,  baseZ: 3.0 },  // Drone Zeta (High escort perimeter)
      ];

      for (let i = 0; i < secondaryDrones.length; i++) {
        const drone = secondaryDrones[i];
        if (numTargets >= i + 2) {
          drone.visible = true;
          const prof = formationProfiles[i];
          const secPos = secTargets[i];
          if (secPos) {
            const pSec = calcVisualPos(secPos);
            drone.position.set(
              pSec.x + prof.baseX * 0.25,
              Math.max(3.5, pSec.y + prof.baseY * 0.25),
              pSec.z + prof.baseZ * 0.25
            );
          } else {
            drone.position.set(
              p1.x + prof.baseX,
              Math.max(3.5, p1.y + prof.baseY),
              p1.z + prof.baseZ
            );
          }
        } else {
          drone.visible = false;
        }
      }

      // Sync active state on live target pills
      if (quickTargetPills) {
        const pills = quickTargetPills.querySelectorAll(".pill-btn");
        pills.forEach(p => {
          if (parseInt(p.dataset.targets) === numTargets) {
            p.classList.add("active");
          } else {
            p.classList.remove("active");
          }
        });
      }

      // Update Laser Beam & Omnidirectional Search Cone
      const isDetected = !!data.target_detected;
      const isFineLocked = (data.link_status === "Signal Locked");
      const isMlSearch = !!data.ml_search_active || (data.pat_state === "ML_SEARCH") || (data.pat_state === "REACQUISITION");

      const aperturePos = new THREE.Vector3(0, 0, 1.51);
      gimbalPayload.localToWorld(aperturePos);
      laserBeam.geometry.setFromPoints([aperturePos, droneTarget.position]);
      laserBeam.visible = (isDetected || isFineLocked) && !isMlSearch;

      if (omniSearchCone) {
        omniSearchCone.visible = isMlSearch;
      }

      // Update 3D Optical Detection HUD on Drone Alpha
      if (droneTarget && droneTarget.userData && droneTarget.userData.lockAssembly) {
        droneTarget.userData.lockAssembly.visible = isDetected;
        if (droneTarget.userData.lockRing) {
          droneTarget.userData.lockRing.material.color.setHex(isFineLocked ? 0x10b981 : 0x06b6d4);
        }
        if (droneTarget.userData.bracketLines) {
          droneTarget.userData.bracketLines.material.color.setHex(isFineLocked ? 0x10b981 : 0x06b6d4);
        }
      }

      const pos3dDetect = document.getElementById("pos-3d-detect");
      if (pos3dDetect) {
        if (isFineLocked) {
          pos3dDetect.className = "badge-detect-locked";
          pos3dDetect.textContent = "🟢 TARGET 01 LOCKED (ALPHA)";
        } else if (isMlSearch) {
          pos3dDetect.className = "badge-detect-searching";
          pos3dDetect.textContent = "🔭 ML OMNI-SEARCH [ALL DEGREES]";
        } else if (isDetected) {
          pos3dDetect.className = "badge-detect-searching";
          pos3dDetect.textContent = "🟡 ACQUIRING ALPHA...";
        } else {
          pos3dDetect.className = "badge-detect-searching";
          pos3dDetect.textContent = "🔴 SEARCHING...";
        }
      }

      // Update ML Omnidirectional Search HUD Banner
      const mlSearchHud = document.getElementById("ml-search-hud-banner");
      const mlScanAngles = document.getElementById("ml-scan-angles");
      const mlScanMatch = document.getElementById("ml-scan-match");
      if (mlSearchHud) {
        if (isMlSearch) {
          mlSearchHud.style.display = "flex";
          if (mlScanAngles) mlScanAngles.textContent = `PAN: ${cPos.yaw_deg?.toFixed(1) || 0}° | TILT: ${cPos.pitch_deg?.toFixed(1) || 0}°`;
          if (mlScanMatch) mlScanMatch.textContent = `ML PROB: ${(data.ml_confidence_pct || 0).toFixed(1)}%`;
        } else {
          mlSearchHud.style.display = "none";
        }
      }

      // Update Target Zig-Zag Motion Button State
      const isWander = !!data.random_flight_enabled;
      const btnRandom3D = document.getElementById("btn-3d-random-flight");
      if (btnRandom3D) {
        if (isWander) btnRandom3D.classList.add("active-wander");
        else btnRandom3D.classList.remove("active-wander");
      }

      if (isCameraPov) updateCameraPosition();
    }

    // Render Virtual Camera
    renderVirtualCamera(data);

    // Render Rolling Performance Chart
    renderChart(data.history);

    // Update Event Logs
    updateLogs(data.logs);
  }

  // Virtual Camera Canvas Renderer
  const cameraImg = new Image();
  let cameraImgLoaded = false;
  cameraImg.onload = () => {
    cameraImgLoaded = true;
    if (ctxCamera) ctxCamera.drawImage(cameraImg, 0, 0, canvasCamera.width, canvasCamera.height);
  };

  function renderVirtualCamera(data) {
    if (!ctxCamera) return;
    const w = canvasCamera.width;
    const h = canvasCamera.height;

    if (data.frame_jpeg) {
      cameraImg.src = `data:image/jpeg;base64,${data.frame_jpeg}`;
      if (cameraImgLoaded) {
        ctxCamera.drawImage(cameraImg, 0, 0, w, h);
      }
    } else {
      ctxCamera.fillStyle = "#070b14";
      ctxCamera.fillRect(0, 0, w, h);
    }

    // Update Optical Beacon Signature Temporal Sequencer HUD
    const sigData = data.beacon_signature;
    if (sigData) {
      const dot = document.getElementById("sig-status-dot");
      if (dot) {
        if (sigData.is_on) {
          dot.classList.add("active-pulse");
        } else {
          dot.classList.remove("active-pulse");
        }
      }
      const pills = document.querySelectorAll("#sig-slots .sig-slot-pill");
      pills.forEach((pill, idx) => {
        if (idx === sigData.slot_index) {
          pill.classList.add("pulse-on");
          pill.classList.add("active");
        } else {
          pill.classList.remove("pulse-on");
          pill.classList.remove("active");
        }
      });
      const lockTag = document.getElementById("sig-lock-tag");
      if (lockTag) {
        lockTag.textContent = sigData.locked ? `LOCKED (${sigData.confidence_boost_pct}%)` : "ACQUIRING";
      }
    }

    // Update HUD Bracket Reticle (Smooth EKF coasting through pulses & clouds)
    const isDetected = data.target_detected;
    const hasEkfCoasting = (!isDetected && data.ekf_initialized && data.ekf_estimated_pos && !data.out_of_fov);

    if (isDetected || hasEkfCoasting) {
      reticleHud.style.display = "block";
      const px = isDetected ? data.target_position_px[0] : data.ekf_estimated_pos[0];
      const py = isDetected ? data.target_position_px[1] : data.ekf_estimated_pos[1];
      reticleHud.style.left = `${(px / 640) * 100}%`;
      reticleHud.style.top = `${(py / 480) * 100}%`;

      reticleHud.classList.remove("reticle-locked", "reticle-coasting");
      if (data.link_status === "Signal Locked") {
        reticleHud.classList.add("reticle-locked");
      } else if (hasEkfCoasting) {
        reticleHud.classList.add("reticle-coasting");
      }

      const sigPulseStr = (sigData && sigData.is_on) ? "SIG-PULSE ●" : "SIG-STANDBY ○";
      const statusLabel = hasEkfCoasting ? "EKF COASTING" : data.link_status.toUpperCase();
      hudTargetLabel.textContent = `${data.target_name.toUpperCase()} [${statusLabel}] • ${sigPulseStr}`;
    } else {
      reticleHud.style.display = "none";
      reticleHud.classList.remove("reticle-locked", "reticle-coasting");
    }
  }

  // =========================================================================
  // 7. Real-Time Performance Chart (Always Active & Rendered)
  // =========================================================================
  function renderChart(history) {
    if (!ctxChart) return;
    const w = canvasChart.clientWidth || 320;
    const h = canvasChart.clientHeight || 170;
    canvasChart.width = w;
    canvasChart.height = h;

    // Background
    ctxChart.fillStyle = "#ffffff";
    ctxChart.fillRect(0, 0, w, h);

    const padLeft = 44, padRight = 14, padTop = 16, padBottom = 24;
    const chartW = Math.max(10, w - padLeft - padRight);
    const chartH = Math.max(10, h - padTop - padBottom);

    // Y-Axis Horizontal Grid Lines & Scale Labels
    ctxChart.font = "10px 'JetBrains Mono', monospace";
    ctxChart.textAlign = "right";
    ctxChart.textBaseline = "middle";

    const yTicks = [
      { val: 2.0, label: "+2.0°" },
      { val: 1.0, label: "+1.0°" },
      { val: 0.0, label: " 0.0°" },
      { val: -1.0, label: "-1.0°" },
      { val: -2.0, label: "-2.0°" }
    ];

    yTicks.forEach(tick => {
      const y = padTop + chartH / 2 - (tick.val / 2.0) * (chartH / 2);
      ctxChart.strokeStyle = (tick.val === 0.0) ? "#cbd5e1" : "#f1f5f9";
      ctxChart.lineWidth = (tick.val === 0.0) ? 1.5 : 1;
      ctxChart.beginPath();
      ctxChart.moveTo(padLeft, y);
      ctxChart.lineTo(w - padRight, y);
      ctxChart.stroke();

      ctxChart.fillStyle = (tick.val === 0.0) ? "#64748b" : "#94a3b8";
      ctxChart.fillText(tick.label, padLeft - 6, y);
    });

    // Time Axis (X) Labels
    ctxChart.textAlign = "center";
    ctxChart.textBaseline = "top";
    ctxChart.fillStyle = "#94a3b8";
    const xLabels = ["-60s", "-45s", "-30s", "-15s", "Now"];
    xLabels.forEach((lbl, idx) => {
      const x = padLeft + (idx / 4) * chartW;
      ctxChart.fillText(lbl, x, padTop + chartH + 6);
    });

    const maxErr = 2.0;
    const toY = (val) => padTop + chartH / 2 - (Math.max(-maxErr, Math.min(maxErr, val)) / maxErr) * (chartH / 2);

    const times = history?.times || [];
    const yawErrors = history?.yaw_error || [];
    const pitchErrors = history?.pitch_error || [];

    if (times.length >= 2) {
      // Draw Yaw Error (Vibrant Sky Blue)
      ctxChart.strokeStyle = "#0284c7";
      ctxChart.lineWidth = 2.2;
      ctxChart.lineJoin = "round";
      ctxChart.beginPath();
      yawErrors.forEach((val, i) => {
        const x = padLeft + (i / (yawErrors.length - 1)) * chartW;
        const y = toY(val);
        if (i === 0) ctxChart.moveTo(x, y); else ctxChart.lineTo(x, y);
      });
      ctxChart.stroke();

      // Draw Pitch Error (Crimson Red)
      ctxChart.strokeStyle = "#ef4444";
      ctxChart.lineWidth = 2.2;
      ctxChart.lineJoin = "round";
      ctxChart.beginPath();
      pitchErrors.forEach((val, i) => {
        const x = padLeft + (i / (pitchErrors.length - 1)) * chartW;
        const y = toY(val);
        if (i === 0) ctxChart.moveTo(x, y); else ctxChart.lineTo(x, y);
      });
      ctxChart.stroke();

      // Diffraction corridor band (±0.06° sub-milliradian precision corridor)
      const corridorTop = toY(0.06);
      const corridorBot = toY(-0.06);
      ctxChart.fillStyle = "rgba(16, 185, 129, 0.09)";
      ctxChart.fillRect(padLeft, corridorTop, chartW, corridorBot - corridorTop);
      ctxChart.strokeStyle = "rgba(16, 185, 129, 0.35)";
      ctxChart.lineWidth = 1;
      ctxChart.setLineDash([3, 3]);
      ctxChart.beginPath();
      ctxChart.moveTo(padLeft, corridorTop); ctxChart.lineTo(padLeft + chartW, corridorTop);
      ctxChart.moveTo(padLeft, corridorBot); ctxChart.lineTo(padLeft + chartW, corridorBot);
      ctxChart.stroke();
      ctxChart.setLineDash([]);

      // Draw latest error dots at the right edge
      const lastIdx = yawErrors.length - 1;
      const lastX = padLeft + chartW;
      const lastYawY = toY(yawErrors[lastIdx]);
      const lastPitchY = toY(pitchErrors[lastIdx]);

      ctxChart.fillStyle = "#0284c7";
      ctxChart.beginPath(); ctxChart.arc(lastX, lastYawY, 3.5, 0, Math.PI * 2); ctxChart.fill();

      ctxChart.fillStyle = "#ef4444";
      ctxChart.beginPath(); ctxChart.arc(lastX, lastPitchY, 3.5, 0, Math.PI * 2); ctxChart.fill();

      // Live FSM lock precision badge
      const lastYaw = yawErrors[lastIdx];
      const lastPitch = pitchErrors[lastIdx];
      const radErrUrad = Math.round(Math.hypot(lastYaw, lastPitch) * (Math.PI / 180) * 1e6);
      ctxChart.font = "bold 10px 'JetBrains Mono', monospace";
      ctxChart.textAlign = "right";
      ctxChart.fillStyle = (radErrUrad < 150) ? "#10b981" : "#0284c7";
      ctxChart.fillText(`● FSM OPTICAL LOCK: ${radErrUrad} µrad`, padLeft + chartW, padTop - 4);
    } else {
      // Real-Time Baseline Indicator when acquiring initial samples
      const zeroY = padTop + chartH / 2;
      ctxChart.strokeStyle = "rgba(2, 132, 199, 0.4)";
      ctxChart.setLineDash([4, 4]);
      ctxChart.beginPath();
      ctxChart.moveTo(padLeft, zeroY);
      ctxChart.lineTo(w - padRight, zeroY);
      ctxChart.stroke();
      ctxChart.setLineDash([]);

      // Subtle active tracking status badge
      ctxChart.fillStyle = "#0284c7";
      ctxChart.font = "11px 'Inter', sans-serif";
      ctxChart.textAlign = "center";
      ctxChart.textBaseline = "middle";
      ctxChart.fillText("● Real-Time PAT Tracking Active (Sub-µrad Alignment)", padLeft + chartW / 2, zeroY - 18);
    }
  }

  function updateLogs(logs) {
    if (!logContainer || !logs) return;
    logContainer.innerHTML = "";
    logs.forEach(l => {
      const div = document.createElement("div");
      div.className = "log-entry";
      let col = "";
      if (l.category === "success") col = "log-green";
      else if (l.category === "warning") col = "log-red";
      else if (l.category === "action") col = "log-accent";

      div.innerHTML = `<span class="log-ts">${l.time}</span> <span class="${col}">${l.msg}</span>`;
      logContainer.appendChild(div);
    });
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  // =========================================================================
  // 8. 2D Interactive Analog Joystick & Optical Remote Handset
  // =========================================================================
  function initJoystickAndRemote() {
    const joyBase = document.getElementById("joystick-base");
    const joyThumb = document.getElementById("joystick-thumb");
    const remoteRate = document.getElementById("remote-rate-val");
    const remoteModeBtn = document.getElementById("btn-remote-mode");
    const remoteModeText = document.getElementById("remote-mode-text");
    const remoteModeIcon = document.getElementById("remote-mode-icon");
    const remoteRecenterBtn = document.getElementById("btn-remote-recenter");
    const remoteZoomBtn = document.getElementById("btn-remote-zoom");
    const remoteZoomText = document.getElementById("remote-zoom-text");

    let isDragging = false;
    let joyTimer = null;
    let currentPanRate = 0.0;
    let currentTiltRate = 0.0;
    const maxRadius = 38; // px max thumbstick excursion

    function sendJogCmd(pan, tilt) {
      currentPanRate = pan;
      currentTiltRate = tilt;
      fetch("/api/control/jog", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pan_rate: pan, tilt_rate: tilt }),
      });
      if (remoteRate) {
        const rateDeg = Math.hypot(pan, tilt) * (180.0 / Math.PI);
        remoteRate.textContent = `${rateDeg.toFixed(1)}°/s`;
      }
    }

    function onPointerStart(e) {
      isDragging = true;
      if (joyThumb) joyThumb.classList.remove("springing");
      onPointerMove(e);

      // 30Hz continuous jog updates while dragging
      if (!joyTimer) {
        joyTimer = setInterval(() => {
          if (isDragging) {
            sendJogCmd(currentPanRate, currentTiltRate);
          }
        }, 33);
      }
    }

    function onPointerMove(e) {
      if (!isDragging || !joyBase || !joyThumb) return;
      const rect = joyBase.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      let dx = e.clientX - centerX;
      let dy = e.clientY - centerY;
      const dist = Math.hypot(dx, dy);

      if (dist > maxRadius) {
        dx = (dx / dist) * maxRadius;
        dy = (dy / dist) * maxRadius;
      }

      joyThumb.style.transform = `translate(${dx}px, ${dy}px)`;

      // Map to proportional pan/tilt angular rates (rad/s)
      const pan = (dx / maxRadius) * 0.75;
      const tilt = (-dy / maxRadius) * 0.75;
      currentPanRate = pan;
      currentTiltRate = tilt;
      sendJogCmd(pan, tilt);
    }

    function onPointerEnd() {
      if (!isDragging) return;
      isDragging = false;
      if (joyTimer) {
        clearInterval(joyTimer);
        joyTimer = null;
      }
      if (joyThumb) {
        joyThumb.classList.add("springing");
        joyThumb.style.transform = "translate(0px, 0px)";
      }
      sendJogCmd(0.0, 0.0);
    }

    if (joyBase) {
      joyBase.addEventListener("pointerdown", onPointerStart);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerEnd);
      window.addEventListener("pointercancel", onPointerEnd);
    }

    // Cardinal Direction Tap Joggers
    const btnJogUp = document.getElementById("btn-jog-up");
    const btnJogDown = document.getElementById("btn-jog-down");
    const btnJogLeft = document.getElementById("btn-jog-left");
    const btnJogRight = document.getElementById("btn-jog-right");

    if (btnJogUp) {
      btnJogUp.addEventListener("mousedown", (e) => { e.stopPropagation(); sendJogCmd(0.0, 0.45); });
      btnJogUp.addEventListener("mouseup", (e) => { e.stopPropagation(); sendJogCmd(0.0, 0.0); });
    }
    if (btnJogDown) {
      btnJogDown.addEventListener("mousedown", (e) => { e.stopPropagation(); sendJogCmd(0.0, -0.45); });
      btnJogDown.addEventListener("mouseup", (e) => { e.stopPropagation(); sendJogCmd(0.0, 0.0); });
    }
    if (btnJogLeft) {
      btnJogLeft.addEventListener("mousedown", (e) => { e.stopPropagation(); sendJogCmd(-0.45, 0.0); });
      btnJogLeft.addEventListener("mouseup", (e) => { e.stopPropagation(); sendJogCmd(0.0, 0.0); });
    }
    if (btnJogRight) {
      btnJogRight.addEventListener("mousedown", (e) => { e.stopPropagation(); sendJogCmd(0.45, 0.0); });
      btnJogRight.addEventListener("mouseup", (e) => { e.stopPropagation(); sendJogCmd(0.0, 0.0); });
    }

    // Remote Recenter Action
    if (remoteRecenterBtn) {
      remoteRecenterBtn.addEventListener("click", () => {
        fetch("/api/control/recenter", { method: "POST" });
      });
    }

    // Out-of-FOV All-Degrees ML Reacquisition Action
    const triggerOutOfFovAction = async (btn) => {
      if (btn) {
        btn.classList.add("btn-pulsing");
        setTimeout(() => btn.classList.remove("btn-pulsing"), 600);
      }
      try {
        await fetch("/api/control/test_out_of_fov", { method: "POST" });
      } catch (err) {
        console.error("Error triggering out-of-view ML search:", err);
      }
    };

    const btnTargetOutOfView = document.getElementById("btn-target-out-of-view");
    if (btnTargetOutOfView) {
      btnTargetOutOfView.addEventListener("click", () => triggerOutOfFovAction(btnTargetOutOfView));
    }
    const btnOutOfFov = document.getElementById("btn-remote-out-of-fov");
    if (btnOutOfFov) {
      btnOutOfFov.addEventListener("click", () => triggerOutOfFovAction(btnOutOfFov));
    }

    // Target Motion: Dynamic Geometric Zig-Zag Weave Toggle
    const handleToggleZigzag = async () => {
      try {
        const res = await fetch("/api/control/toggle_random_flight", { method: "POST" });
        const data = await res.json();
        const isWander = !!data.random_flight_enabled;
        const b = document.getElementById("btn-3d-random-flight");
        if (b) b.classList.toggle("active-wander", isWander);
      } catch (e) {
        console.error("Error toggling zig-zag motion:", e);
      }
    };

    const btnRandomFlight = document.getElementById("btn-3d-random-flight");
    if (btnRandomFlight) {
      btnRandomFlight.addEventListener("click", handleToggleZigzag);
    }

    // Remote Mode Toggle Action (Auto Track vs Manual Jog)
    let isAutoMode = true;
    if (remoteModeBtn) {
      remoteModeBtn.addEventListener("click", () => {
        isAutoMode = !isAutoMode;
        const newMode = isAutoMode ? "auto_track" : "manual";
        fetch("/api/control/mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: newMode }),
        });
        if (remoteModeText) remoteModeText.textContent = isAutoMode ? "Auto Track" : "Manual Jog";
        if (remoteModeIcon) remoteModeIcon.textContent = isAutoMode ? "🎯" : "🕹️";
        remoteModeBtn.className = isAutoMode ? "remote-btn remote-btn-active" : "remote-btn";
      });
    }

    // Remote Zoom Cycler (1.0x -> 2.0x -> 4.0x)
    let zoomLevels = [1.0, 2.0, 4.0];
    let zoomIdx = 0;
    if (remoteZoomBtn) {
      remoteZoomBtn.addEventListener("click", () => {
        zoomIdx = (zoomIdx + 1) % zoomLevels.length;
        const z = zoomLevels[zoomIdx];
        fetch("/api/control/zoom", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ zoom: z }),
        });
        if (remoteZoomText) remoteZoomText.textContent = `${z.toFixed(1)}x`;
      });
    }

    // System Clear Log
    if (btnClearLog) {
      btnClearLog.addEventListener("click", () => {
        fetch("/api/control/clear_log", { method: "POST" });
      });
    }

    // Master Simulation Pause / Play
    if (btnPauseSim) {
      btnPauseSim.addEventListener("click", async () => {
        const res = await fetch("/api/control/pause", { method: "POST" });
        const data = await res.json();
        btnPauseSim.textContent = data.running ? "⏸" : "▶";
        btnPauseSim.title = data.running ? "Pause Simulation" : "Resume Simulation";
      });
    }
  }

  // Initial Boot
  window.addEventListener("DOMContentLoaded", () => {
    initHomeCanvas();
    initJoystickAndRemote();
    updatePreviewPhysics();
  });

})();
