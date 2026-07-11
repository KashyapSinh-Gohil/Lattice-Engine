"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import {
  Activity, Cpu, Sliders, Gauge, MessageSquareCode, Sun, Moon,
  Zap, MapPin, Shield, LayoutGrid, Droplets, BarChart3, Send,
  ChevronRight, Layers, RefreshCw, Search, ToggleLeft, ToggleRight
} from "lucide-react";

const API = "";

type Domain = "grid" | "agro";

export default function LatticeDashboard() {
  // ─── Core State ───
  const [loading, setLoading] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [domain, setDomain] = useState<Domain>("grid");
  const [heroVisible, setHeroVisible] = useState(true);

  // ─── Grid State ───
  const [gridSys, setGridSys] = useState<any>(null);
  const [gridTimings, setGridTimings] = useState<any>({});
  const [feeders, setFeeders] = useState<any[]>([]);
  const [selectedFeederId, setSelectedFeederId] = useState<string | null>(null);
  const [transformers, setTransformers] = useState<any[]>([]);

  // ─── Agro State ───
  const [agroSys, setAgroSys] = useState<any>(null);
  const [agroTimings, setAgroTimings] = useState<any>({});
  const [villages, setVillages] = useState<any[]>([]);
  const [selectedVillageId, setSelectedVillageId] = useState<string | null>(null);
  const [triggers, setTriggers] = useState<any[]>([]);

  // ─── Shared State ───
  const [activeTab, setActiveTab] = useState<string>("primary");
  const [freshness, setFreshness] = useState<number>(0);
  const [blockClock, setBlockClock] = useState<string>("--:--");
  const [benchData, setBenchData] = useState<any>(null);

  // ─── What-If / Allocation ───
  const [wiMw, setWiMw] = useState<string>("40");
  const [wiBudget, setWiBudget] = useState<string>("40");
  const [wiN, setWiN] = useState<string>("50000");
  const [wiOut, setWiOut] = useState<any>(null);
  const [wiLoading, setWiLoading] = useState(false);

  // ─── Pipeline Re-run ───
  const [rerunEngine, setRerunEngine] = useState<string>("cpu");
  const [rerunning, setRerunning] = useState(false);

  // ─── Copilot Chat ───
  const [chatInput, setChatInput] = useState("");
  const [chatMsgs, setChatMsgs] = useState<any[]>([
    { role: "bot", text: "Lattice Copilot online. Select a domain above and ask me anything about the live pipeline data.", tools: [] },
  ]);
  const [chatLoading, setChatLoading] = useState(false);

  // ─── DOM Refs ───
  const mapRef = useRef<any>(null);
  const mapInstanceRef = useRef<any>(null);
  const layerGroupRef = useRef<any>(null);
  const lastDomainRef = useRef<string | null>(null);
  const chartCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<any>(null);
  const msgsEndRef = useRef<HTMLDivElement | null>(null);

  // ═══════════════════════════════════════════════
  //  Theme Management
  // ═══════════════════════════════════════════════
  useEffect(() => {
    const isDark = localStorage.getItem("theme") === "dark" ||
      (!("theme" in localStorage) && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setDarkMode(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  const toggleTheme = () => {
    const next = !darkMode;
    setDarkMode(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  // ═══════════════════════════════════════════════
  //  Data Loading — Both Domains
  // ═══════════════════════════════════════════════
  useEffect(() => {
    async function loadAll() {
      const t0 = Date.now();
      try {
        // Grid data
        const gRes = await fetch(`${API}/api/summary`);
        if (gRes.ok) {
          const g = await gRes.json();
          setGridSys(g.system);
          setGridTimings(g.timings);
          if (g.system?.deficit_mw) setWiMw(String(Math.max(5, Math.round(g.system.deficit_mw))));
        }
        const fRes = await fetch(`${API}/api/feeders`);
        if (fRes.ok) setFeeders(await fRes.json());
        const tRes = await fetch(`${API}/api/transformers`);
        if (tRes.ok) setTransformers(await tRes.json());

        // Agro data
        const aRes = await fetch(`${API}/api/agro/summary`);
        if (aRes.ok) {
          const a = await aRes.json();
          setAgroSys(a.system);
          setAgroTimings(a.timings);
          if (a.system?.water_need_total_ml) setWiBudget(String(Math.max(10, Math.round(a.system.water_need_total_ml * 0.33))));
        }
        const vRes = await fetch(`${API}/api/agro/villages`);
        if (vRes.ok) setVillages(await vRes.json());
        const trRes = await fetch(`${API}/api/agro/triggers`);
        if (trRes.ok) setTriggers(await trRes.json());

        // Benchmarks
        const bRes = await fetch(`${API}/api/benchmarks`);
        if (bRes.ok) setBenchData(await bRes.json());
      } catch (err) {
        console.error("Lattice init error:", err);
      } finally {
        const elapsed = Date.now() - t0;
        setTimeout(() => setLoading(false), Math.max(0, 1000 - elapsed));
      }
    }
    loadAll();
  }, []);

  // ═══════════════════════════════════════════════
  //  Freshness Timer
  // ═══════════════════════════════════════════════
  useEffect(() => {
    if (loading) return;
    let age = 0;
    const iv = setInterval(() => { age++; setFreshness(age); }, 1000);
    const ck = setInterval(() => {
      const now = new Date();
      const s = 900 - ((now.getMinutes() % 15) * 60 + now.getSeconds());
      setBlockClock(`${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`);
    }, 1000);
    return () => { clearInterval(iv); clearInterval(ck); };
  }, [loading]);

  // ═══════════════════════════════════════════════
  //  Map Rendering — responds to domain switch
  // ═══════════════════════════════════════════════
  useEffect(() => {
    if (loading || typeof window === "undefined") return;
    const hasData = domain === "grid" ? feeders.length > 0 : villages.length > 0;
    if (!hasData) return;

    import("leaflet").then((L) => {
      // Create or reuse map
      const mapEl = document.getElementById("lattice-map");
      if (!mapEl) return;

      if (!mapInstanceRef.current) {
        const map = L.map(mapEl, { zoomControl: false, attributionControl: false })
          .setView([23.0300, 72.5800], 11);
        L.tileLayer(darkMode
          ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
          { attribution: "© OpenStreetMap", maxZoom: 19 }).addTo(map);
        mapInstanceRef.current = map;
        layerGroupRef.current = L.layerGroup().addTo(map);
      }

      const map = mapInstanceRef.current;
      const lg = layerGroupRef.current;
      lg.clearLayers();

      const getPolygonPoints = (lat: number, lon: number, r: number, idx: number, sides: number = 7) => {
        const hash = (n: number) => { let h = Math.sin(n) * 43758.5453123; return h - Math.floor(h); };
        const numSides = sides === 8 ? 5 + Math.floor(hash(idx) * 3) : sides;
        const pts: [number, number][] = [];
        let currentAngle = hash(idx) * Math.PI; 
        for (let i = 0; i < numSides; i++) {
          const angleStep = (2 * Math.PI) / numSides;
          const angleJitter = (hash(idx + i) - 0.5) * (angleStep * 0.5);
          currentAngle += angleStep + angleJitter;
          const rad = r * (0.6 + 0.6 * hash(idx * 10 + i)); 
          pts.push([
            lat + (rad / 111000) * Math.cos(currentAngle),
            lon + (rad / (111000 * Math.cos((lat * Math.PI) / 180))) * Math.sin(currentAngle)
          ]);
        }
        return pts;
      };

      const isDomainSwitch = lastDomainRef.current !== domain;
      lastDomainRef.current = domain;

      if (domain === "grid") {
        const fciColor = (f: number) => f > 0.8 ? "#ef4444" : f > 0.5 ? "#f59e0b" : "#10b981"; // Vibrant Tailwind colors
        feeders.slice(0, 400).forEach((f, idx) => {
          const isSel = selectedFeederId === f.feeder_id;
          const poly = L.polygon(getPolygonPoints(f.lat, f.lon, 500 + f.current_mw * 90, idx, 6), {
            color: isSel ? (darkMode ? "#ffffff" : "#000000") : (darkMode ? "#444444" : "#aaaaaa"), weight: isSel ? 3.5 : 1,
            fillColor: fciColor(f.fci), fillOpacity: isSel ? 0.9 : 0.6,
          });
          poly.bindTooltip(`<b>${f.name}</b> (${f.feeder_id})<br/>FCI: <b>${f.fci.toFixed(2)}</b><br/>Load: ${f.current_mw.toFixed(1)} MW<br/>Outages: ${f.failure_history?.length || 0}${f.protected_class ? "<br/><b style='color:#d9383a'>PROTECTED</b>" : ""}`, { direction: "top", opacity: 0.95 });
          poly.on("click", () => { setSelectedFeederId(f.feeder_id); });
          lg.addLayer(poly);
          if (f.protected_class) {
            lg.addLayer(L.marker([f.lat, f.lon], { icon: L.divIcon({ className: "", html: `<div style="font-size:10px;font-weight:bold;color:#d9383a">★</div>`, iconSize: [12, 12] }) }));
          }
        });
        if (isDomainSwitch) map.setView([23.0300, 72.5800], 11); // Center on Ahmedabad for grid
      } else {
        const vapiColor = (v: number) => v > 0.6 ? "#ef4444" : v > 0.38 ? "#f59e0b" : "#3b82f6"; // Blue for well-watered, red for high priority
        villages.slice(0, 600).forEach((v, idx) => {
          const isSel = selectedVillageId === v.village_id;
          const poly = L.polygon(getPolygonPoints(v.lat, v.lon, 300 + Math.sqrt(v.area_ha) * 35, idx, 8), {
            color: isSel ? (darkMode ? "#ffffff" : "#000000") : (darkMode ? "#444444" : "#aaaaaa"), weight: isSel ? 3.5 : 1.2,
            fillColor: vapiColor(v.vapi), fillOpacity: isSel ? 0.85 : 0.6,
          });
          poly.bindTooltip(`<b>${v.name}</b> (${v.village_id})<br/>VAPI: <b>${v.vapi.toFixed(2)}</b> · Reach: ${v.canal_reach}<br/>Yield: ${v.yield_pred}/${v.normal_yield} t/ha${v.insurance_trigger ? "<br/><b style='color:#d9383a'>INSURANCE TRIGGER-HIT</b>" : ""}`, { direction: "top", opacity: 0.95 });
          poly.on("click", () => { setSelectedVillageId(v.village_id); });
          lg.addLayer(poly);
          if (v.canal_reach === "tail") {
            lg.addLayer(L.marker([v.lat, v.lon], { icon: L.divIcon({ className: "", html: `<div style="font-size:9px;font-weight:bold;color:#000;background:#fff;border:1px solid #000;padding:0 4px;border-radius:3px;box-shadow:2px 2px 0 #000">TAIL</div>`, iconSize: [30, 10] }) }));
          }
        });
        if (isDomainSwitch) map.setView([22.0, 70.9], 8); // Broader view covering multiple Saurashtra districts
      }
    });
  }, [loading, domain, feeders, villages, selectedFeederId, selectedVillageId, darkMode]);

  // ═══════════════════════════════════════════════
  //  Chart Rendering — responds to domain switch
  // ═══════════════════════════════════════════════
  useEffect(() => {
    if (loading || !chartCanvasRef.current) return;
    const sys = domain === "grid" ? gridSys : agroSys;
    if (!sys) return;

    const txtColor = darkMode ? "#ffffff" : "#000000";
    const dimColor = darkMode ? "#a0a0a0" : "#555555";
    const gridColor = darkMode ? "#333333" : "#e0e0e0";
    const chartBg = darkMode ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)";

    let chartInstance: any;
    import("chart.js/auto").then(({ Chart }) => {
      if (chartRef.current) chartRef.current.destroy();

      if (domain === "grid") {
        const hist = sys.system_load?.map((p: any) => ({ x: p.ts.slice(11, 16), y: p.mw })) || [];
        const fc = sys.system_forecast?.map((p: any, i: number) => ({ x: (sys.forecast_ts?.[i] || p.ts).slice(11, 16), y: p.mw })) || [];
        const temps = sys.temperature?.map((p: any) => ({ x: p.ts.slice(11, 16), y: p.c })) || [];
        const labels = [...hist.map((p: any) => p.x), ...fc.map((p: any) => p.x)];

        chartInstance = new Chart(chartCanvasRef.current as any, {
          type: "line",
          data: {
            labels,
            datasets: [
              { label: "Load MW", data: hist.map((p: any) => p.y), borderColor: txtColor, backgroundColor: chartBg, fill: true, pointRadius: 0, borderWidth: 3, tension: 0.3 },
              { label: "Forecast MW", data: [...Array(hist.length - 1).fill(null), hist.at(-1)?.y, ...fc.map((p: any) => p.y)], borderColor: dimColor, borderDash: [6, 4], pointRadius: 0, borderWidth: 2.5, tension: 0.3 },
              { label: "Supply cap", data: Array(labels.length).fill(sys.supply_cap_mw), borderColor: "#d9383a", borderDash: [3, 4], pointRadius: 0, borderWidth: 2 },
              { label: "Temp C", yAxisID: "y1", data: temps.slice(-hist.length).map((p: any) => p.y), borderColor: "#8b5cf6", pointRadius: 0, borderWidth: 1.5 },
            ],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { labels: { color: txtColor, boxWidth: 14, font: { family: "JetBrains Mono", size: 11 } } } },
            scales: {
              x: { ticks: { color: txtColor, maxTicksLimit: 12, font: { family: "JetBrains Mono" } }, grid: { color: gridColor } },
              y: { ticks: { color: txtColor, font: { family: "JetBrains Mono" } }, grid: { color: gridColor }, title: { display: true, text: "MW", color: txtColor, font: { family: "JetBrains Mono" } } },
              y1: { position: "right", ticks: { color: "#8b5cf6", font: { family: "JetBrains Mono" } }, grid: { display: false } },
            },
          },
        } as any);
      } else {
        const nd = sys.ndvi_curve || [];
        const rn = sys.rain_curve || [];
        const labels = nd.map((p: any) => p.date.slice(5));
        chartInstance = new Chart(chartCanvasRef.current as any, {
          type: "line",
          data: {
            labels,
            datasets: [
              { label: "Mean NDVI", data: nd.map((p: any) => p.ndvi), borderColor: txtColor, backgroundColor: chartBg, fill: true, pointRadius: 0, borderWidth: 2, tension: 0.3, yAxisID: "y" },
              { label: "Rainfall (mm)", type: "bar", data: rn.map((p: any) => p.rain), backgroundColor: "rgba(85,85,85,0.45)", borderWidth: 1.5, borderColor: txtColor, yAxisID: "y1" } as any,
            ],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { labels: { color: txtColor, font: { size: 10, weight: "bold", family: "monospace" } } } },
            scales: {
              x: { ticks: { color: dimColor, font: { size: 9, family: "monospace" } }, grid: { color: gridColor } },
              y: { min: 0, max: 1.0, ticks: { color: dimColor, font: { size: 9, family: "monospace" } }, grid: { color: gridColor }, title: { display: true, text: "NDVI", color: txtColor, font: { size: 10, weight: "bold" } } },
              y1: { position: "right", ticks: { color: dimColor, font: { size: 9, family: "monospace" } }, grid: { display: false }, title: { display: true, text: "Rainfall (mm)", color: txtColor, font: { size: 10, weight: "bold" } } },
            },
          },
        } as any);
      }
      chartRef.current = chartInstance;
    });

    return () => { if (chartInstance) chartInstance.destroy(); };
  }, [loading, domain, gridSys, agroSys, darkMode]);

  // Scroll chat
  useEffect(() => { msgsEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMsgs]);

  // ═══════════════════════════════════════════════
  //  Handlers
  // ═══════════════════════════════════════════════
  const switchDomain = (d: Domain) => {
    if (d === domain) return;
    setDomain(d);
    setActiveTab("primary");
    setWiOut(null);
    setSelectedFeederId(null);
    setSelectedVillageId(null);
    setHeroVisible(false);
  };

  const handleRunWhatIf = async () => {
    setWiLoading(true);
    try {
      if (domain === "grid") {
        const res = await fetch(`${API}/api/whatif`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target_mw: Number(wiMw), n_candidates: Number(wiN) }) });
        if (res.ok) setWiOut(await res.json());
      } else {
        const res = await fetch(`${API}/api/agro/allocate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ budget_ml: parseFloat(wiBudget), n_candidates: parseInt(wiN) }) });
        if (res.ok) setWiOut(await res.json());
      }
    } catch (err) { console.error(err); }
    finally { setWiLoading(false); }
  };

  const handleRerun = async () => {
    setRerunning(true);
    try {
      const d = domain === "agro" ? "agro" : "grid";
      await fetch(`${API}/api/pipeline/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ engine: rerunEngine, domain: d, data: d === "agro" ? "data/agro_1m" : "data/city_1m" }) });
      const iv = setInterval(async () => {
        const s = await (await fetch(`${API}/api/pipeline/status?domain=${d}`)).json();
        if (!s.running) { clearInterval(iv); setRerunning(false); window.location.reload(); }
      }, 3000);
    } catch (err) { console.error(err); setRerunning(false); }
  };

  const handleSendChat = async (text?: string) => {
    const msg = (text || chatInput).trim();
    if (!msg) return;
    setChatInput("");
    setChatMsgs(prev => [...prev, { role: "user", text: msg }]);
    setChatLoading(true);
    try {
      const res = await fetch(`${API}/api/agent`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: msg, domain, history: chatMsgs.map(m => ({ role: m.role, text: m.text })) }) });
      if (res.ok) {
        const d = await res.json();
        setChatMsgs(prev => [...prev, { role: "bot", text: d.reply, tools: d.tool_calls || d.tools || [] }]);
      }
    } catch (err) { setChatMsgs(prev => [...prev, { role: "bot", text: `Error: ${err}` }]); }
    finally { setChatLoading(false); }
  };

  const formatChat = (s: string) => (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\*(.+?)\*/g, "<i>$1</i>");

  // ═══════════════════════════════════════════════
  //  Derived values
  // ═══════════════════════════════════════════════
  const sys = domain === "grid" ? gridSys : agroSys;
  const timings = domain === "grid" ? gridTimings : agroTimings;

  const gridTabs = [
    { id: "primary", icon: <Activity className="w-4 h-4" />, label: "Shed Priority" },
    { id: "tx", icon: <Cpu className="w-4 h-4" />, label: "Transformers" },
    { id: "wi", icon: <Sliders className="w-4 h-4" />, label: "What-If" },
    { id: "chat", icon: <MessageSquareCode className="w-4 h-4" />, label: "Copilot" },
  ];

  const agroTabs = [
    { id: "primary", icon: <MapPin className="w-4 h-4" />, label: "Advisory Priority" },
    { id: "trig", icon: <Shield className="w-4 h-4" />, label: "Insurance Triggers" },
    { id: "wi", icon: <Droplets className="w-4 h-4" />, label: "Allocate Water" },
    { id: "chat", icon: <MessageSquareCode className="w-4 h-4" />, label: "Copilot" },
  ];

  const tabs = domain === "grid" ? gridTabs : agroTabs;

  // ═══════════════════════════════════════════════
  //  Loading Screen
  // ═══════════════════════════════════════════════
  if (loading) {
    return (
      <div className="fixed inset-0 bg-panel flex flex-col items-center justify-center z-[99999]">
        <div className="w-16 h-6 border-4 border-line bg-transparent rounded-full animate-morph" />
        <div className="mt-7 font-mono text-[11px] tracking-[3px] font-bold text-center animate-pulse-opacity text-txt">
          LATTICE // INITIALIZING
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════
  //  Grid KPIs
  // ═══════════════════════════════════════════════
  const gridKPIs = [
    { label: "System Load", val: gridSys?.current_mw?.toLocaleString(), sub: "MW right now" },
    { label: "Forecast Peak", val: gridSys?.forecast_peak_mw?.toLocaleString(), sub: "MW 4h XGBoost" },
    { label: "Supply Deficit", val: gridSys?.deficit_mw > 0 ? `${gridSys.deficit_mw}` : "0", sub: "MW vs allocation", color: gridSys?.deficit_mw > 0 ? "border-[#d9383a] shadow-[4px_4px_0px_#d9383a]" : "", valColor: gridSys?.deficit_mw > 0 ? "text-[#d9383a]" : "" },
    { label: "Headroom", val: gridSys ? Math.round(gridSys.capacity_mw - gridSys.forecast_peak_mw)?.toLocaleString() : "—", sub: "MW to capacity", valColor: "text-[#2b9348]" },
    { label: "TX Watchlist", val: gridSys?.tx_watchlist, sub: "high-risk DTs", color: gridSys?.tx_watchlist > 0 ? "border-[#d9383a] shadow-[4px_4px_0px_#d9383a]" : "", valColor: gridSys?.tx_watchlist > 0 ? "text-[#d9383a]" : "" },
    { label: "Risk AUC", val: gridSys?.risk_model_auc, sub: "XGBoost 90d" },
  ];

  const agroKPIs = [
    { label: "Villages", val: agroSys?.villages?.toLocaleString(), sub: "under advisory" },
    { label: "Area at Risk", val: agroSys?.at_risk_pct !== undefined ? `${agroSys.at_risk_pct}%` : "—", sub: "% of hectares", color: agroSys?.at_risk_pct > 30 ? "border-[#d9383a] shadow-[4px_4px_0px_#d9383a]" : "", valColor: agroSys?.at_risk_pct > 30 ? "text-[#d9383a]" : "" },
    { label: "Triggers Hit", val: agroSys?.insurance_triggers?.toLocaleString(), sub: "PMFBY-style", color: agroSys?.insurance_triggers > 0 ? "border-[#f2a104] shadow-[4px_4px_0px_#f2a104]" : "", valColor: agroSys?.insurance_triggers > 0 ? "text-[#f2a104]" : "" },
    { label: "Yield Forecast", val: agroSys?.mean_yield_pred ? `${agroSys.mean_yield_pred}/${agroSys.mean_normal_yield}` : "—", sub: "t/ha vs normal" },
    { label: "Yield Saveable", val: agroSys?.yield_saveable_total_t?.toLocaleString(), sub: "t with timely water", valColor: "text-[#2b9348]" },
    { label: "Loss AUC", val: agroSys?.risk_model_auc, sub: "XGBoost 5 seasons" },
  ];

  const kpis = domain === "grid" ? gridKPIs : agroKPIs;

  // ═══════════════════════════════════════════════
  //  RENDER
  // ═══════════════════════════════════════════════
  return (
    <div className="min-h-screen text-txt">
      {/* ─────── Hero Section (collapsible) ─────── */}
      {heroVisible && (
        <section className="max-w-[1100px] mx-auto px-7 pt-12 pb-8 animate-tab-fade">
          <div className="flex justify-between items-start gap-4 flex-wrap w-full">
            <div>
              <h1 className="text-[40px] md:text-[52px] font-extrabold tracking-tight text-txt font-sans leading-none flex items-center gap-3">
                <Layers className="w-10 h-10 md:w-12 md:h-12" strokeWidth={2.5} />
                Lattice
              </h1>
              <p className="text-[13px] md:text-[14px] font-semibold uppercase tracking-wider text-dim mt-2">
                High-Performance Resource Allocation Engine
              </p>
            </div>
            <button onClick={toggleTheme} className="font-mono text-[10.5px] border-2 border-line bg-panel text-txt px-3 py-1.5 rounded-lg font-bold hover:opacity-85 cursor-pointer shadow-solid-sm active:translate-y-[1px] active:shadow-none transition-all flex items-center gap-1.5">
              {darkMode ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
              <span>{darkMode ? "LIGHT" : "DARK"}</span>
            </button>
          </div>

          <p className="mt-6 text-[19px] md:text-[22px] font-bold text-txt max-w-[840px] leading-snug">
            One engine for two climate-stressed lifelines:{" "}
            <span className="underline decoration-2 decoration-line">power</span> and{" "}
            <span className="underline decoration-2 decoration-line">food</span>.
          </p>
          <p className="mt-3 text-[13px] text-dim max-w-[840px] leading-relaxed">
            Under climate stress a scarce resource must be allocated across many units, each carrying
            a risk score and a fairness history. Lattice turns large-scale sensor data into an explainable,
            fairness-aware allocation refreshed inside the operating window.
          </p>

          {/* Domain Selector Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-8">
            <button onClick={() => switchDomain("grid")} className={`group text-left bg-panel border-2 rounded-[20px] p-5 shadow-solid hover:shadow-solid-lg hover:translate-y-[-2px] transition-all duration-200 ${domain === "grid" ? "border-[#000] ring-2 ring-offset-2 ring-black dark:ring-white" : "border-line"}`}>
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-5 h-5" />
                <h2 className="text-[18px] font-black text-txt">Power Grid</h2>
              </div>
              <p className="text-[12px] text-dim leading-relaxed">Peak-demand shedding, transformer failure prediction, feeder criticality index, fairness-aware load dispatch.</p>
              <div className="flex gap-1.5 flex-wrap mt-3">
                {["Feeder Criticality", "TX Risk", "Shed What-If"].map(s => (
                  <span key={s} className="font-mono text-[9px] font-bold bg-panel border border-line px-2 py-0.5 rounded-full text-txt">{s}</span>
                ))}
              </div>
              <div className="mt-3 font-bold text-[12px] text-txt group-hover:translate-x-1 transition-transform inline-flex items-center gap-1">
                {domain === "grid" ? "Active" : "Switch"} <ChevronRight className="w-3.5 h-3.5" />
              </div>
            </button>

            <button onClick={() => switchDomain("agro")} className={`group text-left bg-panel border-2 rounded-[20px] p-5 shadow-solid hover:shadow-solid-lg hover:translate-y-[-2px] transition-all duration-200 ${domain === "agro" ? "border-[#000] ring-2 ring-offset-2 ring-black dark:ring-white" : "border-line"}`}>
              <div className="flex items-center gap-2 mb-2">
                <Droplets className="w-5 h-5" />
                <h2 className="text-[18px] font-black text-txt">Agriculture</h2>
              </div>
              <p className="text-[12px] text-dim leading-relaxed">Village-level crop stress advisory, insurance trigger detection, canal water allocation with tail-reach fairness bonus.</p>
              <div className="flex gap-1.5 flex-wrap mt-3">
                {["VAPI Priority", "Insurance Triggers", "Water Allocation"].map(s => (
                  <span key={s} className="font-mono text-[9px] font-bold bg-panel border border-line px-2 py-0.5 rounded-full text-txt">{s}</span>
                ))}
              </div>
              <div className="mt-3 font-bold text-[12px] text-txt group-hover:translate-x-1 transition-transform inline-flex items-center gap-1">
                {domain === "agro" ? "Active" : "Switch"} <ChevronRight className="w-3.5 h-3.5" />
              </div>
            </button>
          </div>

          {/* Stack tags */}
          <div className="mt-8">
            <div className="font-mono text-[10px] text-faint tracking-widest uppercase font-bold mb-2">Connected Stack</div>
            <div className="flex gap-1.5 flex-wrap">
              {["Cloud Storage", "BigQuery", "Cloud Run", "Vertex AI", "Dataproc Spark", "XGBoost", "FastAPI", "Pandas", "Scikit-Learn"].map((s, i) => (
                <span key={i} className="font-mono text-[10px] border-2 border-line bg-panel px-2.5 py-0.5 rounded-lg text-txt font-bold">{s}</span>
              ))}
            </div>
          </div>

          {/* Collapse hero button */}
          <button onClick={() => setHeroVisible(false)} className="mt-6 font-mono text-[10px] text-dim hover:text-txt transition-colors flex items-center gap-1">
            Collapse overview <ChevronRight className="w-3 h-3 rotate-90" />
          </button>
        </section>
      )}

      {/* ─────── Dashboard ─────── */}
      <div className="flex flex-col lg:flex-row gap-5 max-w-[1650px] mx-auto px-5 pb-8">
        {/* Sidebar */}
        <aside className="w-full lg:w-[280px] shrink-0 bg-panel border-2 border-line rounded-[24px] p-5 flex flex-col gap-4 shadow-solid lg:h-[calc(100vh-40px)] lg:sticky lg:top-5">
          <div>
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-2 text-[18px] font-black tracking-tight font-sans text-txt">
                <Layers className="w-5 h-5" strokeWidth={2.5} />
                Lattice
              </div>
              <button onClick={toggleTheme} className="font-mono text-[9px] border border-line bg-panel text-txt px-2 py-0.5 rounded font-bold hover:opacity-80 cursor-pointer shadow-solid-sm active:translate-y-[1px] active:shadow-none transition-all flex items-center gap-1">
                {darkMode ? <Sun className="w-3 h-3" /> : <Moon className="w-3 h-3" />}
              </button>
            </div>

            {/* Domain Toggle */}
            <div className="flex rounded-xl border-2 border-line overflow-hidden mb-4">
              <button onClick={() => switchDomain("grid")} className={`flex-1 py-2 text-[11px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-1.5 ${domain === "grid" ? "bg-txt text-panel" : "bg-panel text-dim hover:text-txt"}`}>
                <Zap className="w-3.5 h-3.5" /> Grid
              </button>
              <button onClick={() => switchDomain("agro")} className={`flex-1 py-2 text-[11px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-1.5 ${domain === "agro" ? "bg-txt text-panel" : "bg-panel text-dim hover:text-txt"}`}>
                <Droplets className="w-3.5 h-3.5" /> Agro
              </button>
            </div>

            {!heroVisible && (
              <button onClick={() => setHeroVisible(true)} className="font-mono text-[9px] text-dim hover:text-txt mb-2 flex items-center gap-1">
                <ChevronRight className="w-3 h-3 -rotate-90" /> Show overview
              </button>
            )}
          </div>

          <div className="h-px bg-line opacity-15" />

          {/* Tab Selection */}
          <nav className="flex flex-col gap-1.5 flex-grow">
            {tabs.map(t => (
              <button key={t.id} onClick={() => { setActiveTab(t.id); setWiOut(null); }}
                className={`flex items-center gap-2.5 px-3 py-2.5 text-[12.5px] font-semibold border-2 rounded-xl transition-all duration-200 ${activeTab === t.id ? "bg-txt border-line text-panel shadow-solid-sm" : "bg-transparent border-transparent text-dim hover:bg-neutral-100 dark:hover:bg-neutral-900 hover:text-txt"}`}>
                {t.icon} <span>{t.label}</span>
              </button>
            ))}
          </nav>

          {/* Sidebar Footer Stats */}
          <div className="flex flex-col gap-3 mt-auto">
            <div className="flex items-center gap-2 border-2 border-line px-3 py-2 rounded-xl text-[10px] font-bold uppercase font-mono text-txt">
              <span className={`w-2 h-2 rounded-full animate-pulse-dot ${sys ? "bg-[#2b9348] shadow-[0_0_6px_#2b9348]" : "bg-[#f2a104] shadow-[0_0_6px_#f2a104]"}`} />
              <span>{sys ? "live" : "no artifacts"}</span>
              <span className="ml-auto text-faint">{domain.toUpperCase()}</span>
            </div>
            <div className="flex flex-col gap-0.5 text-[9.5px] font-mono text-dim">
              <div className="flex justify-between"><span>ENGINE</span><b className="text-txt uppercase">{timings?.engine || "--"}</b></div>
              <div className="flex justify-between"><span>DATA AGE</span><b className="text-txt">{freshness < 120 ? `${freshness}s` : `${Math.round(freshness / 60)}m`}</b></div>
              {domain === "grid" && <div className="flex justify-between"><span>BLOCK T-</span><b className="text-txt">{blockClock}</b></div>}
              {domain === "grid" && <div className="flex justify-between"><span>OUTDOOR</span><b className="text-txt">{gridSys?.temp_now_c ?? "--"}C</b></div>}
              {domain === "agro" && <div className="flex justify-between"><span>SEASON</span><b className="text-txt">WK {agroSys?.ndvi_curve?.length || "--"}</b></div>}
            </div>

            <div className="pt-2 border-t border-line/10 text-[8.5px] font-sans text-dim leading-tight">
              <b className="text-txt block mb-0.5">DATA POLICY COMPLIANCE</b>
              Anonymized under DPDP Act 2023. Telemetry aggregated at feeder/transformer and canal-gate levels. No PII. Conforms to NDSAP guidelines.
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-grow min-w-0 flex flex-col gap-4">
          {/* Alert Banner */}
          {domain === "grid" && gridSys?.deficit_mw > 0 && (
            <div className="flex items-center gap-3 px-4 py-3 bg-[#fff0f0] border-2 border-[#d9383a] rounded-[16px] shadow-[3px_3px_0px_#d9383a] animate-tab-fade">
              <span className="font-mono font-bold text-[11px] text-[#d9383a]">GRID EMERGENCY</span>
              <span className="text-[12.5px] text-black">Peak <b>{gridSys.forecast_peak_mw} MW</b> exceeds supply <b>{gridSys.supply_cap_mw} MW</b> — need <b>{gridSys.deficit_mw} MW</b> relief.</span>
              <button onClick={() => setActiveTab("wi")} className="ml-auto bg-[#d9383a] border-2 border-black text-white font-bold px-3 py-1.5 rounded-full text-[11px] uppercase hover:bg-black transition-all">PLAN RELIEF</button>
            </div>
          )}
          {domain === "agro" && agroSys?.insurance_triggers > 0 && (
            <div className="flex items-center gap-3 px-4 py-3 bg-[#fff0f0] border-2 border-[#d9383a] rounded-[16px] shadow-[3px_3px_0px_#d9383a] animate-tab-fade">
              <span className="font-mono font-bold text-[11px] text-[#d9383a]">DRY-SPELL STRESS</span>
              <span className="text-[12.5px] text-black"><b>{agroSys.insurance_triggers}</b> villages crossed triggers. <b>{agroSys.yield_saveable_total_t?.toLocaleString()} t</b> saveable.</span>
              <button onClick={() => setActiveTab("wi")} className="ml-auto bg-[#d9383a] border-2 border-black text-white font-bold px-3 py-1.5 rounded-full text-[11px] uppercase hover:bg-black transition-all">ALLOCATE</button>
            </div>
          )}

          {/* KPIs */}
          <div className="font-mono text-[9.5px] text-faint tracking-widest uppercase font-bold">
            LATTICE // {domain.toUpperCase()} // OVERVIEW
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2.5">
            {kpis.map((k: any, idx: number) => (
              <div key={idx} className={`bg-panel border-2 border-line rounded-[16px] p-3 shadow-solid hover:translate-y-[-1px] hover:shadow-solid-lg transition-all duration-200 ${k.color || ""}`}>
                <div className="font-mono text-[9px] uppercase font-bold text-dim">{k.label}</div>
                <div className={`font-sans text-[22px] font-extrabold mt-0.5 ${k.valColor || "text-txt"}`}>{k.val ?? "--"}</div>
                <div className="text-[10px] text-faint mt-0.5">{k.sub}</div>
              </div>
            ))}
          </div>

          {/* Main 2-Column Layout */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {/* Left: Map + Chart + Pipeline */}
            <div className="flex flex-col gap-4">
              <div className="bg-panel border-2 border-line rounded-[20px] p-4 shadow-solid hover:translate-y-[-1px] hover:shadow-solid-lg transition-all">
                <div className="font-mono text-[9.5px] text-faint tracking-widest uppercase font-bold mb-1.5">
                  LATTICE // {domain === "grid" ? "FEEDER NETWORK" : "VILLAGE NETWORK"}
                </div>
                <h3 className="font-sans font-extrabold text-[14px] uppercase border-b-2 border-line pb-2 mb-3 flex items-center gap-2 text-txt">
                  <span>{domain === "grid" ? "Feeder Coverage Map" : "Village Coverage Map"}</span>
                  <span className="ml-auto text-[9px] font-mono normal-case text-dim">
                    {domain === "grid" ? <>FCI: <span className="text-[#2b9348] font-bold">low</span> — <span className="text-[#d9383a] font-bold">critical</span> | <span style={{color:"#8b5cf6"}}>◆</span> protected</> : <>VAPI: <span className="text-[#2b9348] font-bold">low</span> — <span className="text-[#d9383a] font-bold">critical</span> | TAIL = tail-reach</>}
                  </span>
                </h3>
                <div id="lattice-map" className="w-full h-[340px] rounded-xl border-2 border-line" />
              </div>

              <div className="bg-panel border-2 border-line rounded-[20px] p-4 shadow-solid hover:translate-y-[-1px] hover:shadow-solid-lg transition-all">
                <div className="font-mono text-[9.5px] text-faint tracking-widest uppercase font-bold mb-1.5">
                  LATTICE // {domain === "grid" ? "LOAD METRICS" : "NDVI + RAINFALL"}
                </div>
                <h3 className="font-sans font-extrabold text-[14px] uppercase border-b-2 border-line pb-2 mb-3 text-txt">
                  {domain === "grid" ? "Load Forecast Trend" : "Crop Health Trend"}
                </h3>
                <div className="h-[240px]"><canvas ref={chartCanvasRef} /></div>
              </div>


            </div>

            {/* Right: Tab Panel */}
            <div className="bg-panel border-2 border-line rounded-[20px] p-4 shadow-solid min-h-[500px] flex flex-col">
              <div className="font-mono text-[9.5px] text-faint tracking-widest uppercase font-bold mb-1.5">
                LATTICE // OPERATIONS
              </div>
              <h3 className="font-sans font-extrabold text-[14px] uppercase border-b-2 border-line pb-2 mb-4 text-txt flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> Operations Desk
              </h3>

              {/* ─── TAB: Primary List (Grid Feeders or Agro Villages) ─── */}
              {activeTab === "primary" && domain === "grid" && (
                <div className="animate-tab-fade overflow-y-auto max-h-[600px]">
                  {feeders.slice(0, 50).map(f => (
                    <div key={f.feeder_id} onClick={() => { setSelectedFeederId(prev => prev === f.feeder_id ? null : f.feeder_id); if (mapInstanceRef.current) mapInstanceRef.current.setView([f.lat, f.lon], 13); }}
                      className={`p-3 border-2 border-line rounded-xl mb-2 cursor-pointer transition-all duration-200 hover:translate-x-[-1px] hover:shadow-solid-sm ${selectedFeederId === f.feeder_id ? "bg-txt text-panel" : "bg-panel text-txt"}`}>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-[10px] text-faint">#{f.rank}</span>
                        <span className="font-sans font-extrabold text-[13px]">{f.name} <span className="font-mono text-[10px] font-normal opacity-60 ml-1">{f.feeder_id}</span></span>
                        <span className="flex-grow h-1.5 bg-neutral-200 dark:bg-neutral-800 border border-line rounded-full overflow-hidden mx-1.5 max-w-[100px]">
                          <i className="block h-full bg-gradient-to-r from-[#2b9348] via-[#f2a104] to-[#d9383a]" style={{ width: `${f.fci * 100}%` }} />
                        </span>
                        <span className="font-mono font-bold text-[12px] w-9 text-right" style={{ color: selectedFeederId === f.feeder_id ? "inherit" : f.fci > 0.66 ? "#d9383a" : f.fci > 0.4 ? "#f2a104" : "#2b9348" }}>{f.fci.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                        {f.reason_codes.map((c: string) => {
                          let cls = "bg-panel text-txt border-line";
                          if (c.startsWith("PROTECTED")) cls = selectedFeederId === f.feeder_id ? "bg-[#221133] text-[#dd88ff] border-[#dd88ff]" : "bg-[#f3e8ff] text-[#8b5cf6] border-[#8b5cf6]";
                          else if (["OVERLOAD-4H", "AT-LIMIT", "RECENT-SHED", "SAGS", "TX-RISK"].some(x => c.startsWith(x))) cls = selectedFeederId === f.feeder_id ? "bg-[#551111] text-[#ff8888] border-[#ff8888]" : "bg-[#fff0f0] text-[#d9383a] border-[#d9383a]";
                          else if (c === "FAIR-OK") cls = selectedFeederId === f.feeder_id ? "bg-[#113311] text-[#88ff88] border-[#88ff88]" : "bg-[#f0fff4] text-[#2b9348] border-[#2b9348]";
                          return <span key={c} className={`font-mono text-[8px] font-bold px-1.5 py-0.5 rounded-full border ${cls}`}>{c}</span>;
                        })}
                        <span className="font-mono text-[9px] text-dim ml-auto">{f.current_mw.toFixed(1)} MW</span>
                      </div>
                      {selectedFeederId === f.feeder_id && f.components && (
                        <div className="bg-white/5 border border-white/10 p-2.5 rounded-lg mt-2 text-[11px] flex flex-col gap-0.5 animate-tab-fade">
                          {Object.entries(f.components).map(([k, v]: any) => (
                            <div key={k} className="flex items-center gap-2"><span className="text-dim w-24 capitalize">{k.replace("_", " ")}</span><span className="flex-grow h-1 bg-white/10 border border-white/20 rounded-full overflow-hidden"><i className="block h-full bg-white" style={{ width: `${v * 100}%` }} /></span><span className="font-mono w-8 text-right">{v.toFixed(2)}</span></div>
                          ))}
                          <div className="text-[9px] text-dim/70 border-t border-white/10 pt-1 mt-0.5">Fairness: {f.shed_hours_30d.toFixed(1)}h shed | {f.complaints_30d} complaints | {f.customers.toLocaleString()} customers</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {activeTab === "primary" && domain === "agro" && (
                <div className="animate-tab-fade overflow-y-auto max-h-[600px]">
                  {villages.slice(0, 50).map(v => (
                    <div key={v.village_id} onClick={() => { setSelectedVillageId(prev => prev === v.village_id ? null : v.village_id); if (mapInstanceRef.current) mapInstanceRef.current.setView([v.lat, v.lon], 11); }}
                      className={`p-3 border-2 border-line rounded-xl mb-2 cursor-pointer transition-all duration-200 hover:translate-x-[-1px] hover:shadow-solid-sm ${selectedVillageId === v.village_id ? "bg-txt text-panel" : "bg-panel text-txt"}`}>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-[10px] text-faint">#{v.rank}</span>
                        <span className="font-sans font-extrabold text-[13px]">{v.name} <span className="font-mono text-[10px] font-normal opacity-60 ml-1">{v.village_id}</span></span>
                        <span className="flex-grow h-1.5 bg-neutral-200 dark:bg-neutral-800 border border-line rounded-full overflow-hidden mx-1.5 max-w-[100px]">
                          <i className="block h-full bg-gradient-to-r from-[#2b9348] via-[#f2a104] to-[#d9383a]" style={{ width: `${v.vapi * 100}%` }} />
                        </span>
                        <span className="font-mono font-bold text-[12px] w-9 text-right" style={{ color: selectedVillageId === v.village_id ? "inherit" : v.vapi > 0.6 ? "#d9383a" : v.vapi > 0.38 ? "#f2a104" : "#2b9348" }}>{v.vapi.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center gap-1.5 mt-1.5 flex-wrap text-[9px] font-mono text-dim">
                        <span className={`font-bold px-1.5 py-0.5 rounded-full border ${v.canal_reach === "tail" ? "bg-[#fff0f0] text-[#d9383a] border-[#d9383a]" : v.canal_reach === "mid" ? "bg-[#fffbe6] text-[#f2a104] border-[#f2a104]" : "bg-[#f0fff4] text-[#2b9348] border-[#2b9348]"}`}>{v.canal_reach.toUpperCase()}</span>
                        {v.insurance_trigger && <span className="font-bold bg-[#fff0f0] text-[#d9383a] border border-[#d9383a] px-1.5 py-0.5 rounded-full">TRIGGER-HIT</span>}
                        <span className="ml-auto">{v.yield_pred}/{v.normal_yield} t/ha | {v.area_ha} ha</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* ─── TAB: Transformers (Grid only) ─── */}
              {activeTab === "tx" && domain === "grid" && (
                <div className="animate-tab-fade overflow-x-auto">
                  <table className="min-w-full">
                    <thead><tr><th>DT ID</th><th>Feeder</th><th>72h Fail Risk</th><th>SHAP Codes</th><th>Peak Load</th><th>Age</th></tr></thead>
                    <tbody>
                      {transformers.slice(0, 40).map(t => (
                        <tr key={t.transformer_id} className="hover:bg-neutral-50 dark:hover:bg-neutral-900">
                          <td className="font-mono font-bold text-txt">{t.transformer_id}</td>
                          <td className="font-mono text-dim">{t.feeder_id}</td>
                          <td>
                            <span className="inline-block w-[60px] h-1.5 bg-neutral-200 border border-line rounded-full overflow-hidden mr-2"><i className="block h-full bg-gradient-to-r from-[#f2a104] to-[#d9383a]" style={{ width: `${t.p_fail_72h * 100}%` }} /></span>
                            <span className="font-mono font-bold" style={{ color: t.p_fail_72h > 0.5 ? "#d9383a" : "#f2a104" }}>{(t.p_fail_72h * 100).toFixed(0)}%</span>
                          </td>
                          <td><div className="flex gap-1">{t.reason_codes.map((c: string) => <span key={c} className="font-mono text-[8px] font-bold bg-[#fff0f0] text-[#d9383a] border border-[#d9383a] px-1.5 py-0.5 rounded-full">{c}</span>)}</div></td>
                          <td className="font-mono">{(t.loading_max * 100).toFixed(0)}%</td>
                          <td className="font-mono">{t.age_years}y</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* ─── TAB: Insurance Triggers (Agro only) ─── */}
              {activeTab === "trig" && domain === "agro" && (
                <div className="animate-tab-fade overflow-y-auto max-h-[600px]">
                  {triggers.length === 0 ? <p className="text-dim text-[12px]">No insurance triggers in current dataset.</p> :
                    triggers.slice(0, 50).map((t: any, idx: number) => (
                      <div key={idx} className="p-3 border-2 border-line rounded-xl mb-2 bg-panel">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-[10px] text-[#d9383a]">TRIGGER</span>
                          <span className="font-sans font-bold text-[13px] text-txt">{t.name || t.village_id}</span>
                          <span className="ml-auto font-mono text-[10px] text-dim">{t.type || "NDVI-drop"}</span>
                        </div>
                        <div className="text-[10px] font-mono text-dim mt-1">
                          NDVI dropped {t.ndvi_drop?.toFixed ? t.ndvi_drop.toFixed(2) : "--"} | Area: {t.area_ha} ha | Reach: {t.canal_reach}
                        </div>
                      </div>
                    ))
                  }
                </div>
              )}

              {/* ─── TAB: What-If / Allocation ─── */}
              {activeTab === "wi" && (
                <div className="animate-tab-fade flex flex-col gap-3">
                  <div className="flex gap-2 items-center flex-wrap">
                    <label className="font-mono text-[9px] font-bold uppercase text-dim">{domain === "grid" ? "Relief Target" : "Water Budget"}</label>
                    <input type="number" value={domain === "grid" ? wiMw : wiBudget} onChange={e => domain === "grid" ? setWiMw(e.target.value) : setWiBudget(e.target.value)} className="w-20 px-2 py-1 border-2 border-line rounded-lg text-[13px] bg-panel text-txt" />
                    <span className="font-mono font-bold text-[11px] text-txt">{domain === "grid" ? "MW" : "ML"}</span>
                    <label className="font-mono text-[9px] font-bold uppercase text-dim ml-3">Candidates</label>
                    <select value={wiN} onChange={e => setWiN(e.target.value)} className="px-2 py-1 border-2 border-line rounded-lg text-[11px] bg-panel text-txt">
                      <option value="10000">10k</option><option value="50000">50k</option><option value="200000">200k</option><option value="500000">500k</option>
                    </select>
                    <button disabled={wiLoading} onClick={handleRunWhatIf} className="ml-auto bg-txt border-2 border-line text-panel font-extrabold text-[11px] uppercase px-4 py-2 rounded-full hover:opacity-80 transition-all">{wiLoading ? "EVALUATING..." : "EVALUATE"}</button>
                  </div>
                  <div className="text-[10px] text-dim font-mono">{domain === "grid" ? "Hospital/water/transit lines automatically protected." : "Tail-reach fairness bonus applied to under-served villages."}</div>

                  {wiOut && (
                    <div className="flex flex-col gap-3 animate-tab-fade">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {[
                          { label: "Plans evaluated", val: wiOut.plans_evaluated?.toLocaleString() },
                          { label: "Exec Time", val: `${wiOut.eval_seconds}s` },
                          { label: "Throughput", val: `${wiOut.plans_per_second?.toLocaleString()}/s` },
                          { label: "Manual equiv.", val: `${wiOut.equivalent_manual_hours}h` },
                        ].map((s, i) => (
                          <div key={i} className="bg-panel border-2 border-line p-2.5 rounded-xl">
                            <div className="font-mono text-[8px] uppercase font-bold text-dim">{s.label}</div>
                            <div className="font-sans font-extrabold text-[16px] text-txt mt-0.5">{s.val}</div>
                          </div>
                        ))}
                      </div>
                      {wiOut.plans?.map((p: any, idx: number) => (
                        <div key={idx} className={`border-2 border-line p-3 rounded-[14px] ${idx === 0 ? "border-[#2b9348] shadow-[3px_3px_0px_#2b9348]" : "bg-panel"}`}>
                          <div className="flex items-center gap-2 border-b border-line/20 pb-1.5 mb-1.5">
                            <span className="font-sans font-extrabold text-[14px] text-txt">Plan {String.fromCharCode(65 + idx)}</span>
                            {idx === 0 && <span className="font-mono text-[8px] font-bold bg-[#f0fff4] text-[#2b9348] border border-[#2b9348] px-2 py-0.5 rounded-full">RECOMMENDED</span>}
                            <span className="ml-auto font-mono text-[8px] font-bold border border-line bg-panel px-2 py-0.5 rounded-full text-txt">{p.feasible ? "FEASIBLE" : "SHORTFALL"}</span>
                          </div>
                          <div className="flex gap-3 flex-wrap text-[11px] text-dim font-mono">
                            <span>Relief: <b className="text-txt">{p.relief_mw || p.water_ml} {domain === "grid" ? "MW" : "ML"}</b></span>
                            <span>Pain: <b className="text-txt">{p.pain_total}</b></span>
                            <span>Fairness: <b className="text-txt">{p.fairness_penalty}</b></span>
                            <span>Units: <b className="text-txt">{p.n_feeders || p.n_villages}</b></span>
                          </div>
                          <div className="flex flex-wrap gap-1 mt-2">
                            {(p.feeders || p.villages || []).map((u: any) => (
                              <span key={u.name} className="font-mono text-[9px] bg-panel border border-line px-1.5 py-0.5 rounded-full text-txt">{u.name} {u.mw ? `${u.mw}MW` : u.ml ? `${u.ml}ML` : ""}</span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}


              {/* ─── TAB: Copilot Chat ─── */}
              {activeTab === "chat" && (
                <div className="animate-tab-fade flex flex-col flex-grow h-[450px]">
                  <div className="flex-grow overflow-y-auto flex flex-col gap-2.5 pr-2">
                    {chatMsgs.map((m, idx) => (
                      <div key={idx} className={`max-w-[85%] px-3.5 py-2.5 rounded-xl border-2 border-line ${m.role === "user" ? "self-end bg-txt text-panel rounded-br-none" : "self-start bg-panel text-txt rounded-bl-none"}`}>
                        <div className="text-[12.5px] leading-relaxed" dangerouslySetInnerHTML={{ __html: formatChat(m.text) }} />
                        {m.tools?.length > 0 && (
                          <div className="flex gap-1 flex-wrap mt-2">
                            {m.tools.map((t: any, i: number) => <span key={i} className="font-mono text-[8px] font-bold bg-neutral-100 dark:bg-neutral-800 text-txt border border-line px-1.5 py-0.5 rounded-full">{t.tool}</span>)}
                          </div>
                        )}
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="self-start bg-panel text-txt rounded-bl-none max-w-[85%] px-3.5 py-2.5 rounded-xl border-2 border-line">
                        <div className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-txt animate-bounce" /><span className="w-1.5 h-1.5 rounded-full bg-txt animate-bounce [animation-delay:0.2s]" /><span className="w-1.5 h-1.5 rounded-full bg-txt animate-bounce [animation-delay:0.4s]" /></div>
                      </div>
                    )}
                    <div ref={msgsEndRef} />
                  </div>
                  <div className="flex gap-1.5 flex-wrap py-2 border-t-2 border-line mt-2">
                    {(domain === "grid" ? [
                      { label: "Plan 40 MW relief", q: "Give me a shed plan for 40 MW with least customer pain" },
                      { label: "TX watchlist", q: "Which transformers are about to fail and why?" },
                      { label: "Status", q: "System status" },
                    ] : [
                      { label: "Allocate 40 ML", q: "Allocate 40 ML of water with tail-reach fairness" },
                      { label: "Triggers", q: "Which villages crossed insurance triggers?" },
                      { label: "Season status", q: "Season status" },
                    ]).map((s, i) => (
                      <span key={i} onClick={() => handleSendChat(s.q)} className="font-mono text-[9px] border-2 border-line bg-panel hover:bg-txt hover:text-panel text-txt px-2.5 py-1 rounded-full cursor-pointer transition-all">{s.label}</span>
                    ))}
                  </div>
                  <div className="flex gap-2 pt-1">
                    <input type="text" value={chatInput} onChange={e => setChatInput(e.target.value)} placeholder={`Ask about ${domain}...`} onKeyDown={e => e.key === "Enter" && handleSendChat()} className="flex-grow px-3 py-2 border-2 border-line rounded-xl outline-none bg-panel text-txt" />
                    <button onClick={() => handleSendChat()} className="bg-txt border-2 border-line text-panel font-extrabold uppercase px-4 py-2 rounded-full hover:opacity-80 transition-all flex items-center gap-1">
                      <Send className="w-3.5 h-3.5" /> SEND
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Footer */}
          <footer className="mt-4 pt-4 border-t-2 border-line/20 flex flex-col md:flex-row justify-between items-start gap-3">
            <div className="text-[10px] text-dim max-w-[600px] leading-relaxed">
              <b>DATA POLICY COMPLIANCE</b>
              <p className="mt-0.5 text-[9.5px] text-faint">Anonymized under DPDP Act 2023. Telemetry aggregated at feeder/transformer and canal-gate levels. No PII processed. Synthetic baselines per NDSAP guidelines.</p>
            </div>
            <div className="text-[10px] font-mono text-faint shrink-0 text-right">Lattice v2.0 | X-Process-Time Enabled</div>
          </footer>
        </main>
      </div>
    </div>
  );
}
