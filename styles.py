"""
TriCV — feuille de style Streamlit
Charte : aéro-défense, aplats, off-white / noir / rouge.
"""

# Palette — à importer depuis app.py si besoin
COLORS = {
    "bg":          "#F5F3F0",   # off-white chaud
    "surface":     "#FFFFFF",
    "ink":         "#0A0A0A",   # noir profond
    "ink_2":       "#2A2A2A",
    "muted":       "#6A6A66",
    "muted_2":     "#8A8A86",
    "border":      "#EAE7E1",
    "border_soft": "#F2EFE9",
    "accent":      "#C8102E",   # rouge MBDA-like
    "accent_dark": "#A30E25",
    "success":     "#1A6E3C",
    "success_bg":  "#EDF7F0",
    "warn":        "#A85A0B",
    "warn_bg":     "#FDF1E4",
    "error_bg":    "#FCECEE",
}


CSS = """
<style>
/* ── Base : typo Helvetica/Arial, off-white warm ──────────────────────────── */
.stApp {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: #F5F3F0;
    color: #0A0A0A;
}
#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* Layout : réduire les marges par défaut pour un look dashboard dense */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ── Sidebar : noir profond, wordmark rouge, nav épurée ───────────────────── */
[data-testid="stSidebar"] {
    background: #0A0A0A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.5rem;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.2rem 0 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 10.5px !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 700 !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}

/* Sidebar : inputs */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="number"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 4px !important;
    color: #FFFFFF !important;
    font-size: 12.5px !important;
    padding: 0.45rem 0.65rem !important;
}
[data-testid="stSidebar"] input[type="text"]:focus,
[data-testid="stSidebar"] input[type="number"]:focus {
    border-color: #C8102E !important;
    box-shadow: 0 0 0 1px #C8102E !important;
}

/* Sidebar : sliders */
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #C8102E !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 0 2px rgba(200,16,46,0.25) !important;
}

/* Sidebar : captions plus lisibles */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: rgba(255,255,255,0.55) !important;
    font-size: 11.5px !important;
}

/* ── Wordmark TriCV (utilisé dans le header sidebar) ──────────────────────── */
.tricv-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.4rem 0 1.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.tricv-brand-bar {
    width: 3px; height: 26px; background: #C8102E; flex-shrink: 0;
}
.tricv-brand-word {
    font-size: 22px; font-weight: 800; letter-spacing: -0.5px; color: #FFFFFF;
    line-height: 1;
}
.tricv-brand-word .accent { color: #C8102E; }
.tricv-brand-sub {
    margin-top: 8px;
    font-size: 9.5px; letter-spacing: 1.8px; text-transform: uppercase;
    color: rgba(255,255,255,0.38); font-weight: 600;
}

/* Sidebar footer classification */
.tricv-classification {
    margin-top: 1.2rem;
    padding: 0.7rem 0.85rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 2px solid #C8102E;
    font-size: 9.5px;
    letter-spacing: 1.4px;
    color: rgba(255,255,255,0.55) !important;
    font-weight: 700;
    text-transform: uppercase;
}

/* ── Topbar persistant (reste visible quand la sidebar est fermée) ────────── */
.tricv-topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 64px;
    background: rgba(10,10,10,0.95);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(200,16,46,0.25);
    display: flex;
    align-items: center;
    padding: 0 2rem;
    z-index: 999990;
}
.tricv-topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.tricv-topbar-bar {
    width: 3px; height: 24px; background: #C8102E; display: inline-block; border-radius: 2px;
}
.tricv-topbar-word {
    color: #FFFFFF;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1;
}
.tricv-topbar-word .accent { color: #C8102E; }

.stApp { overflow-x: hidden; }

/* Décaler le contenu + la sidebar pour ne pas passer sous la topbar */
.stApp > header { display: none !important; }
.main .block-container { padding-top: 4.5rem !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 4.5rem !important; }

/* ── En-tête principal : bandeau rectangle flottant, délimité rouge ────────── */
.tricv-header {
    position: relative;
    width: 98%;
    max-width: 2200px;
    margin: 0.4rem auto 2.5rem auto;
    border-radius: 12px;
    border: 1px solid rgba(200,16,46,0.5);
    padding: clamp(1.2rem, 2vw, 2rem) clamp(2rem, 5vw, 6rem) clamp(1.5rem, 2.5vw, 2.5rem);
    background:
        radial-gradient(ellipse 55% 80% at 90% 10%, rgba(200,16,46,0.22) 0%, transparent 55%),
        radial-gradient(ellipse 55% 70% at 5% 100%, rgba(200,16,46,0.08) 0%, transparent 60%),
        linear-gradient(135deg, #0A0A0A 0%, #141210 35%, #1E1A18 65%, #2A1618 100%);
    overflow: hidden;
    isolation: isolate;
    box-sizing: border-box;
}
/* Grille de grain subtil pour donner de la matière au dégradé */
.tricv-header::before {
    content: "";
    position: absolute; inset: 0;
    background-image:
        linear-gradient(to right, rgba(255,255,255,0.022) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,0.022) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 50%, black 20%, transparent 80%);
    -webkit-mask-image: radial-gradient(ellipse 80% 60% at 50% 50%, black 20%, transparent 80%);
    pointer-events: none;
    z-index: 0;
}
/* Halo rouge en haut à droite */
.tricv-header::after {
    content: "";
    position: absolute; top: -100px; right: -100px;
    width: 380px; height: 380px; border-radius: 50%;
    background: radial-gradient(circle, rgba(200,16,46,0.28) 0%, transparent 65%);
    filter: blur(14px);
    pointer-events: none;
    z-index: 0;
}
/* Filet lumineux en bas */
.tricv-header .tricv-header-inner {
    position: relative;
    z-index: 1;
    max-width: 1360px;
    margin: 0 auto;
    padding-left: 0.5rem;
}
/* Fondu bas resserré — le bandeau reste un rectangle délimité */
.tricv-header-glow {
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 50px;
    background: linear-gradient(to bottom,
        transparent 0%,
        rgba(210,198,184,0.5) 60%,
        #F5F3F0 100%);
    pointer-events: none;
    z-index: 2;
}
/* Accent rouge subtil juste au-dessus du fondu */
.tricv-header-glow::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 60px;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(200,16,46,0.35) 30%,
        rgba(200,16,46,0.5) 50%,
        rgba(200,16,46,0.35) 70%,
        transparent 100%);
    filter: blur(0.4px);
}
.tricv-header-kicker {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: clamp(5px,0.6vw,8px) clamp(10px,1.2vw,16px) clamp(5px,0.6vw,8px) clamp(8px,1vw,13px);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    color: rgba(255,255,255,0.75) !important;
    font-size: clamp(9px, 0.85vw, 12px);
    font-weight: 700;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    margin-bottom: clamp(0.4rem, 0.8vw, 0.8rem);
    backdrop-filter: blur(8px);
}
.tricv-header-kicker::before {
    content: "";
    display: inline-block;
    width: 6px; height: 6px;
    background: #C8102E;
    border-radius: 50%;
    box-shadow: 0 0 8px #C8102E;
    animation: tricv-pulse 2.4s ease-in-out infinite;
}
@keyframes tricv-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #C8102E; }
    50%      { opacity: 0.55; box-shadow: 0 0 2px #C8102E; }
}
/* Forcer le blanc — surcharge les styles globaux h1/h2/p de .main */
.tricv-header, .tricv-header * { color: #FFFFFF; }
/* Logo TriCV intégré dans le bandeau */
.tricv-header-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: clamp(0.6rem, 1.2vw, 1.2rem);
}
.tricv-header-brand-bar {
    width: 3px;
    height: clamp(18px, 2vw, 28px);
    background: #C8102E;
    flex-shrink: 0;
    border-radius: 2px;
}
.tricv-header-brand-word {
    color: #FFFFFF;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: clamp(1rem, 1.8vw, 2rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1;
}
.tricv-header-brand-word .accent { color: #C8102E; }

h1.tricv-header-title,
.main .tricv-header h1.tricv-header-title,
.tricv-header h1.tricv-header-title {
    color: #FFFFFF !important;
    font-size: clamp(1.3rem, 2.2vw, 2.6rem) !important;
    font-weight: 300 !important;
    letter-spacing: -1.5px !important;
    line-height: 1.05 !important;
    margin: 0 !important;
    text-shadow: 0 2px 30px rgba(0,0,0,0.4);
    text-transform: none !important;
    display: block !important;
}
h1.tricv-header-title::before,
.main .tricv-header h1.tricv-header-title::before { content: none !important; display: none !important; }
.tricv-header-title .strong { font-weight: 700 !important; color: #FFFFFF !important; }
.tricv-header-title .sep {
    display: inline-block;
    color: #C8102E !important;
    font-weight: 300 !important;
    margin: 0 0.4rem;
    transform: translateY(-2px);
}
.tricv-header-sub {
    color: rgba(255,255,255,0.65) !important;
    font-size: clamp(11px, 0.9vw, 13px);
    margin-top: clamp(0.3rem, 0.6vw, 0.6rem);
    letter-spacing: 0.1px;
    max-width: min(640px, 60vw);
    line-height: 1.55;
}

/* ── Métriques : cartes blanches, bordure fine, accent rouge ──────────────── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #EAE7E1;
    border-top: 2px solid #C8102E;
    border-radius: 8px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0A0A0A;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    font-variant-numeric: tabular-nums;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #6A6A66;
    font-weight: 700;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #8A8A86;
    font-size: 11.5px;
    font-weight: 500;
}

/* Progress bar dans le contenu principal */
.main [data-testid="stProgress"] > div > div > div > div {
    background: #C8102E !important;
}
.main [data-testid="stProgress"] > div > div > div {
    background: #EAE7E1 !important;
    height: 6px !important;
    border-radius: 999px !important;
}

/* ── Boutons : rouge plat, pas de dégradé ─────────────────────────────────── */
.stButton > button,
.stDownloadButton > button {
    background: #C8102E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    padding: 0.65rem 1.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 12px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #A30E25 !important;
    transform: translateY(-1px);
}
.stButton > button:focus,
.stDownloadButton > button:focus {
    box-shadow: 0 0 0 3px rgba(200,16,46,0.25) !important;
}

/* Boutons 'secondary' (type='secondary') : noir outline */
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
    border: 1px solid #0A0A0A !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #F5F3F0 !important;
}

/* ── Tabs : minimales, soulignement rouge ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #EAE7E1;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8A8A86;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 0.85rem 1.8rem;
    border: none;
    border-radius: 0;
    transition: color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0A0A0A;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #0A0A0A !important;
    border-bottom: 2px solid #C8102E !important;
    margin-bottom: -1px;
}

/* ── Expander : carte plate blanche ───────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #EAE7E1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    margin-bottom: 0.6rem;
}
[data-testid="stExpander"] summary {
    color: #0A0A0A !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    padding: 0.9rem 1.1rem !important;
    letter-spacing: 0.2px;
}
[data-testid="stExpander"] summary:hover {
    background: #FAF8F4;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0.4rem 1.1rem 1rem !important;
    border-top: 1px solid #F2EFE9;
}

/* Debug expander : version compacte grise */
.tricv-debug-expander [data-testid="stExpander"] summary {
    color: #8A8A86 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600 !important;
}

/* ── Alertes : fonds clairs, accent gauche ────────────────────────────────── */
div.stSuccess {
    background: #EDF7F0 !important;
    border: 1px solid #CFE8D6 !important;
    border-left: 3px solid #1A6E3C !important;
    border-radius: 6px !important;
    color: #1A4D2A !important;
    padding: 0.85rem 1.1rem !important;
    backdrop-filter: none !important;
}
div.stSuccess * { color: #1A4D2A !important; }

div.stWarning {
    background: #FDF1E4 !important;
    border: 1px solid #F4D9B5 !important;
    border-left: 3px solid #A85A0B !important;
    border-radius: 6px !important;
    color: #6E3C07 !important;
    padding: 0.85rem 1.1rem !important;
    backdrop-filter: none !important;
}
div.stWarning * { color: #6E3C07 !important; }

div.stError {
    background: #FCECEE !important;
    border: 1px solid #F3CFD4 !important;
    border-left: 3px solid #C8102E !important;
    border-radius: 6px !important;
    color: #7A0B1C !important;
    padding: 0.85rem 1.1rem !important;
    backdrop-filter: none !important;
}
div.stError * { color: #7A0B1C !important; }

div.stInfo {
    background: #F5F3F0 !important;
    border: 1px solid #EAE7E1 !important;
    border-left: 3px solid #0A0A0A !important;
    border-radius: 6px !important;
    color: #2A2A2A !important;
    padding: 0.85rem 1.1rem !important;
}
div.stInfo * { color: #2A2A2A !important; }

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #EAE7E1;
    border-radius: 8px;
    background: #FFFFFF;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── Inputs / uploader dans la zone principale ────────────────────────────── */
.main input[type="text"],
.main input[type="number"],
.main textarea {
    background: #FFFFFF !important;
    border: 1px solid #EAE7E1 !important;
    border-radius: 6px !important;
    color: #0A0A0A !important;
}
.main input[type="text"]:focus,
.main input[type="number"]:focus,
.main textarea:focus {
    border-color: #C8102E !important;
    box-shadow: 0 0 0 3px rgba(200,16,46,0.12) !important;
}

[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    border: 1px dashed #D9D6D0 !important;
    border-radius: 8px !important;
    padding: 1.4rem !important;
    transition: all 0.15s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #C8102E !important;
    background: #FEFCFC !important;
}
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section span {
    color: #6A6A66 !important;
}
[data-testid="stFileUploader"] button {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    padding: 0.45rem 1rem !important;
}

/* Checkbox */
.main [data-testid="stCheckbox"] label {
    font-size: 13px !important;
    color: #0A0A0A !important;
    font-weight: 500;
}
.main [data-testid="stCheckbox"] [data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"] {
    background: #C8102E !important;
    border-color: #C8102E !important;
}

/* ── Séparateurs ──────────────────────────────────────────────────────────── */
.main hr {
    border: none !important;
    border-top: 1px solid #EAE7E1 !important;
    margin: 2rem 0 !important;
}

/* ── Titres ───────────────────────────────────────────────────────────────── */
.main h1 {
    color: #0A0A0A !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
    letter-spacing: -0.6px !important;
}
.main h2 {
    color: #0A0A0A !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.6px !important;
    margin-top: 1.8rem !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 10px;
}
.main h2::before {
    content: "";
    display: inline-block;
    width: 14px;
    height: 1px;
    background: #C8102E;
}
.main h3 {
    color: #0A0A0A !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: -0.2px !important;
    margin-top: 1.2rem !important;
}
.main h4 {
    color: #0A0A0A !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: -0.1px !important;
}

/* Captions en gris discret */
.main [data-testid="stCaptionContainer"] p {
    color: #8A8A86 !important;
    font-size: 12px !important;
    line-height: 1.5;
}

/* Liens rouges */
.main a { color: #C8102E !important; text-decoration: none; }
.main a:hover { text-decoration: underline; }

/* Markdown body */
.main .stMarkdown p {
    color: #2A2A2A;
    font-size: 13.5px;
    line-height: 1.55;
}
.main .stMarkdown strong { color: #0A0A0A; font-weight: 700; }

/* Divider natif */
.main [data-testid="stDivider"] hr {
    border-top: 1px solid #EAE7E1 !important;
}
</style>
"""


HEADER_HTML = """
<div class="tricv-header">
    <div class="tricv-header-inner">
        <div class="tricv-header-brand">
            <div class="tricv-header-brand-bar"></div>
            <span class="tricv-header-brand-word">Tri<span class="accent">CV</span></span>
        </div>
        <div class="tricv-header-kicker">Outil d'aide au recrutement · Analyse offline</div>
        <h1 class="tricv-header-title">
            Analyse <span class="strong">CV</span><span class="sep">/</span><span class="strong">Offre d'emploi</span>
        </h1>
        <div class="tricv-header-sub">
            Extraction sémantique, scoring de compétences et synthèse RH générée par LLM — tout en restant dans votre périmètre offline.
        </div>
    </div>
    <div class="tricv-header-glow"></div>
</div>
"""


TOPBAR_HTML = """
<div class="tricv-topbar">
    <div class="tricv-topbar-brand">
        <span class="tricv-topbar-bar"></span>
        <span class="tricv-topbar-word">Tri<span class="accent">CV</span></span>
    </div>
</div>
"""


SIDEBAR_BRAND_HTML = """
<div class="tricv-brand">
    <div class="tricv-brand-bar"></div>
    <div>
        <div class="tricv-brand-word">Tri<span class="accent">CV</span></div>
        <div class="tricv-brand-sub">HR Intelligence Suite</div>
    </div>
</div>
"""


SIDEBAR_FOOTER_HTML = """
<div class="tricv-classification">
    ● Internal · Classification C2
</div>
"""
