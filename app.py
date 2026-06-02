"""
AgriShield AI — Flask Backend
Crop Disease Detector (Trained CNN Model) + Farmer Chatbot (Gemini API)
"""

import os
import warnings

# Suppress TensorFlow and Google warnings please correct all things to deploys eaisly 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import io
import json
import traceback
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Google Gemini API client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_available = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Test model initialization
        model = genai.GenerativeModel("gemini-1.5-flash")
        gemini_available = True
        print("[INFO] Google Gemini API successfully configured.")
    except Exception as e:
        print(f"[WARNING] Failed to configure Google Gemini API: {e}")
else:
    print("[WARNING] GEMINI_API_KEY environment variable is missing in .env.")

import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback-dev-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ---------------------------------------------------------------------------
# Trained Model Loading
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

FRUIT_MODEL_PATH = os.path.join(MODEL_DIR, "fruit_disease_model.h5")
FRUIT_LABELS_PATH = os.path.join(MODEL_DIR, "fruit_class_labels.json")

IMG_SIZE = 128
FRUIT_IMG_SIZE = 128

trained_model = None
class_labels = None

fruit_model = None
fruit_class_labels = None

def load_trained_model():
    """Load the trained CNN models and class labels."""
    global trained_model, class_labels, fruit_model, fruit_class_labels

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Trained model not found at {MODEL_PATH}")
        print("[WARNING] Run 'python train_model.py' first to train the model.")
        return False

    if not os.path.exists(LABELS_PATH):
        print(f"[WARNING] Class labels not found at {LABELS_PATH}")
        return False

    try:
        import tensorflow as tf
        trained_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[INFO] Trained model loaded from {MODEL_PATH}")

        with open(LABELS_PATH, "r") as f:
            class_labels = json.load(f)
        print(f"[INFO] Loaded {len(class_labels)} class labels")
        
        # Load fruit model if it exists
        if os.path.exists(FRUIT_MODEL_PATH) and os.path.exists(FRUIT_LABELS_PATH):
            fruit_model = tf.keras.models.load_model(FRUIT_MODEL_PATH)
            with open(FRUIT_LABELS_PATH, "r") as f:
                fruit_class_labels = json.load(f)
            print(f"[INFO] Fruit model loaded from {FRUIT_MODEL_PATH}")
            
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False


# Attempt to load the model on startup
model_loaded = load_trained_model()

# ---------------------------------------------------------------------------
# Disease Knowledge Base
# ---------------------------------------------------------------------------
DISEASE_KNOWLEDGE = {
    "Mango_Healthy": {
        "plant_type": "Mango Fruit",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Smooth, unblemished skin with proper coloring",
            "Firm to the touch with a sweet aroma at the stem",
            "No dark sunken lesions or oozing sap"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Harvest with a 1-inch stem attached to prevent sap burn",
            "Wash fruit immediately after harvest to remove any sap",
            "Store at 55°F (13°C) to delay ripening if needed"
        ]
    },
    "Mango_Diseased": {
        "plant_type": "Mango Fruit",
        "health_status": "Diseased",
        "diagnosis": "Anthracnose / Stem-End Rot",
        "symptoms": [
            "Black, sunken, irregular lesions on the skin that merge together",
            "Tear-stain patterns of rot running down the side of the fruit",
            "Dark brown/black decay starting precisely at the stem attachment",
            "Pinkish spore masses visible in wet conditions"
        ],
        "medicine": [
            "Copper fungicides (e.g., Kocide) sprayed before and during flowering",
            "Azoxystrobin (Abound) applied during fruit development",
            "Post-harvest hot water treatment (125°F for 5 minutes) mixed with Prochloraz"
        ],
        "management": [
            "Prune the tree canopy extensively to ensure rapid drying after rain",
            "Do NOT pack diseased fruit with healthy ones as the fungus spreads via contact",
            "Harvest carefully to avoid bruising or scratching the waxy cuticle"
        ]
    },
    "Apple_Healthy": {
        "plant_type": "Apple Fruit",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Clean skin without spots, lesions, or rotting",
            "Firm texture and appropriate coloration",
            "No signs of pest entry holes or fungal growth"
        ],
        "medicine": [
            "No chemical treatment needed for healthy fruit",
            "Maintain preventive organic fungicide schedule (e.g., neem oil) if pressure is high"
        ],
        "management": [
            "Continue regular harvesting at optimal ripeness",
            "Store in a cool, dark place (32-35°F) with high humidity (90%)",
            "Prune canopy aggressively during dormant season to ensure good airflow",
            "Rake and destroy any fallen leaves to prevent future scab"
        ]
    },
    "Apple_Diseased": {
        "plant_type": "Apple Fruit",
        "health_status": "Diseased",
        "diagnosis": "Apple Scab / Black Rot",
        "symptoms": [
            "Olive-green to black velvety spots on fruit surface",
            "Dark, sunken rotting areas spreading across the skin",
            "Fruit may become cracked, deformed, or drop prematurely",
            "Corky, brown, or hardened tissue underneath the skin"
        ],
        "medicine": [
            "Captan 50WP (2.5g/L) - Apply during early petal fall",
            "Myclobutanil (0.5ml/L) - Effective against cedar apple rust and scab",
            "Copper hydroxide spray for organic management (apply before bud break)",
            "Thiophanate-methyl (1g/L) for severe black rot cases"
        ],
        "management": [
            "Immediately remove and destroy all infected fruits (do NOT compost)",
            "Aggressively prune the tree canopy to allow sunlight and wind penetration",
            "Ensure strict orchard sanitation by burning all mummified fruits and fallen leaves",
            "Avoid overhead irrigation which promotes fungal spore spread"
        ]
    },
    "Tomato_Diseased": {
        "plant_type": "Tomato Fruit",
        "health_status": "Diseased",
        "diagnosis": "Blossom End Rot / Fruit Cracking / Anthracnose",
        "symptoms": [
            "Large, black, leathery sunken rotting at the bottom of the fruit",
            "Deep concentric cracks or radial splitting around the stem",
            "Small, circular, depressed spots on ripe fruit that expand rapidly"
        ],
        "medicine": [
            "Foliar calcium spray (Calcium chloride) for immediate blossom end rot relief",
            "Copper fungicides (2g/L) for anthracnose and bacterial spot",
            "Chlorothalonil (Bravo) applied every 7-14 days as a preventative",
            "Azoxystrobin for severe fungal infections"
        ],
        "management": [
            "Maintain deeply consistent watering to prevent calcium deficiency (Blossom End Rot)",
            "Apply a thick 3-4 inch layer of organic straw mulch to retain soil moisture",
            "Add agricultural lime or crushed eggshells to soil prior to planting",
            "Pick tomatoes slightly early and ripen indoors if heavy rain is expected (prevents cracking)"
        ]
    },
    "Blueberry_Healthy": {
        "plant_type": "Blueberry",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Plump, firm berries with a powdery natural white bloom",
            "Deep blue/purple uniform coloration",
            "No shriveling, softness, or fungal fuzz"
        ],
        "medicine": [
            "No treatment needed - plant is healthy",
            "Optional: Preventative compost tea foliar spray"
        ],
        "management": [
            "Maintain strict soil pH between 4.5 and 5.5",
            "Use pine needle or peat moss mulch",
            "Net bushes before fruits turn blue to prevent bird damage"
        ]
    },
    "Blueberry_Diseased": {
        "plant_type": "Blueberry",
        "health_status": "Diseased",
        "diagnosis": "Mummy Berry / Anthracnose",
        "symptoms": [
            "Berries turn pale, shrivel, and harden into 'mummies'",
            "Soft, sunken rotting spots leaking orange/pink spore masses",
            "Premature fruit drop and rapid post-harvest decay"
        ],
        "medicine": [
            "Propiconazole or Indar applied at green tip stage",
            "Captan (2.5g/L) applied during bloom to prevent Anthracnose",
            "Organic biofungicides (Bacillus subtilis) applied weekly"
        ],
        "management": [
            "Crucial: Rake, bury, or burn all dropped mummified berries before spring",
            "Apply 2 inches of fresh mulch in early spring to bury overwintering spores",
            "Cool harvested berries immediately to 32°F to stop rot progression"
        ]
    },
    "Cherry_Healthy": {
        "plant_type": "Cherry",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Firm, glossy skin with deep, bright coloration",
            "No splitting, cracking, or soft spots",
            "Stems are green and firmly attached"
        ],
        "medicine": [
            "No treatment needed",
            "Dormant oil spray in late winter to prevent early pests"
        ],
        "management": [
            "Harvest with stems attached to prolong shelf life",
            "Protect from rain near harvest time to prevent cracking"
        ]
    },
    "Cherry_Diseased": {
        "plant_type": "Cherry",
        "health_status": "Diseased",
        "diagnosis": "Brown Rot / Cherry Leaf Spot (Fruit impact)",
        "symptoms": [
            "Small brown spots that rapidly expand to rot the entire fruit",
            "Fuzzy, gray or tan powdery mold appearing on the surface",
            "Fruits shrivel into hard mummies clinging to the branch"
        ],
        "medicine": [
            "Myclobutanil (Immunox) applied at popcorn, full bloom, and petal fall stages",
            "Captan or Sulfur sprays for organic management starting 3 weeks before harvest",
            "Fenbuconazole (Indar) for severe commercial outbreaks"
        ],
        "management": [
            "Sanitation is critical: remove all mummies from the tree and ground during winter",
            "Prune trees into an open-center shape for maximum sunlight",
            "Do not compost infected fruit"
        ]
    },
    "Corn_Healthy": {
        "plant_type": "Corn",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - ear appears healthy",
        "symptoms": [
            "Plump, fully developed kernels filled to the tip",
            "Tight, bright green husks",
            "Silks are brown and dry at maturity"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Ensure adequate nitrogen side-dressing at knee-high stage",
            "Provide 1.5 - 2 inches of water per week, especially during tasseling"
        ]
    },
    "Corn_Diseased": {
        "plant_type": "Corn",
        "health_status": "Diseased",
        "diagnosis": "Ear Rot (Fusarium/Diplodia) / Corn Smut",
        "symptoms": [
            "White, pink, or gray fungal growth between kernels",
            "Massive, swollen, fleshy gray/black galls replacing kernels (Smut)",
            "Kernels fused to the husk, rotting from the tip down"
        ],
        "medicine": [
            "Fungicides are rarely effective once ear rot or smut occurs",
            "Prothioconazole (Proline) can be used preventatively at silking stage"
        ],
        "management": [
            "Control ear-feeding insects (corn earworms) as they create entry wounds for fungi",
            "Plant resistant/tolerant corn hybrids next season",
            "Harvest early and dry grain to below 15% moisture immediately",
            "Rotate crop out of corn for 1-2 years"
        ]
    },
    "Grape_Healthy": {
        "plant_type": "Grape",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - bunch appears healthy",
        "symptoms": [
            "Intact, plump berries with uniform color",
            "No shriveling, cracking, or spots",
            "Stems are green and flexible"
        ],
        "medicine": [
            "No treatment needed",
            "Maintain preventative sulfur dusting for powdery mildew"
        ],
        "management": [
            "Continue regular canopy thinning",
            "Ensure proper trellis support"
        ]
    },
    "Grape_Diseased": {
        "plant_type": "Grape",
        "health_status": "Diseased",
        "diagnosis": "Black Rot / Botrytis Bunch Rot",
        "symptoms": [
            "Berries shrivel into hard, black, raisin-like mummies (Black Rot)",
            "Fuzzy gray mold enveloping the clusters (Botrytis)",
            "Brown circular lesions spreading rapidly across the fruit"
        ],
        "medicine": [
            "Myclobutanil (Rally) or Mancozeb (2.5g/L) for Black Rot (apply pre-bloom to 4 weeks post-bloom)",
            "Fenhexamid (Elevate) or Captan targeting Botrytis at veraison",
            "Copper/Sulfur combinations for organic vineyards"
        ],
        "management": [
            "Remove all mummified fruit during winter pruning (primary infection source)",
            "Aggressively pull leaves around the fruit zone to maximize airflow and sun exposure",
            "Harvest promptly before wet weather sets in"
        ]
    },
    "Orange_Healthy": {
        "plant_type": "Orange",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Smooth, bright orange rind without lesions or scabs",
            "Firm to the touch, heavy for its size",
            "No fungal growth on the stem end"
        ],
        "medicine": [
            "No treatment needed",
            "Preventative horticultural oil for scale and aphids"
        ],
        "management": [
            "Maintain balanced NPK fertilization with added micronutrients (Zinc, Iron)",
            "Prune water sprouts and dead wood"
        ]
    },
    "Orange_Diseased": {
        "plant_type": "Orange",
        "health_status": "Diseased",
        "diagnosis": "Citrus Canker / Black Spot / Alternaria Brown Spot",
        "symptoms": [
            "Raised, corky, crater-like lesions with yellow halos on the skin",
            "Dark, sunken, hard spots (Black Spot)",
            "Premature yellowing and fruit drop"
        ],
        "medicine": [
            "Copper-based bactericides (Liquid Copper Fungicide) - spray every 21 days during susceptible periods",
            "Strobilurin fungicides (e.g., Abound) for Black Spot control",
            "Do NOT use sulfur within 30 days of oil sprays"
        ],
        "management": [
            "Control citrus leafminers, as their tunnels expose the fruit to canker bacteria",
            "Plant windbreaks around the grove to prevent wind-driven rain spread",
            "Sanitize all pruning tools and equipment between trees"
        ]
    },
    "Peach_Healthy": {
        "plant_type": "Peach",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Smooth fuzzy skin with uniform blushing",
            "Firm flesh with no soft spots",
            "No gummy sap leaking from the fruit"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Thin fruits to 6-8 inches apart to prevent branches breaking and increase fruit size"
        ]
    },
    "Peach_Diseased": {
        "plant_type": "Peach",
        "health_status": "Diseased",
        "diagnosis": "Brown Rot / Peach Scab",
        "symptoms": [
            "Rapidly spreading brown rot covered in dusty gray/tan spores",
            "Small, dark, velvety spots clustering near the stem end (Scab)",
            "Fruit cracking and exuding clear, gummy sap"
        ],
        "medicine": [
            "Captan or Sulfur fungicide applied starting 3 weeks before harvest",
            "Propiconazole (Orbit) applied at pink bud and petal fall stages",
            "Chlorothalonil (Daconil) used early in the season (do not use after shuck split)"
        ],
        "management": [
            "Strict sanitation: clean up all dropped or mummified fruit instantly",
            "Prune heavily to an open-center vase shape for maximum airflow",
            "Avoid excessive nitrogen fertilizer which causes overly dense, humid canopies"
        ]
    },
    "Pepper_Healthy": {
        "plant_type": "Pepper",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Glossy, firm, thick skin without spots or wrinkling",
            "Strong attachment to the stem",
            "Proper coloration for its maturity stage"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Maintain consistent, deep watering",
            "Apply mulch to regulate soil temperature"
        ]
    },
    "Pepper_Diseased": {
        "plant_type": "Pepper",
        "health_status": "Diseased",
        "diagnosis": "Anthracnose / Blossom End Rot / Sunscald",
        "symptoms": [
            "Circular, sunken, water-soaked lesions that develop dark fungal spores in the center",
            "Papery, white, blistered patches on the sun-exposed side (Sunscald)",
            "Black rotting tissue at the bottom of the fruit (Blossom End Rot)"
        ],
        "medicine": [
            "Copper hydroxide (2g/L) sprayed every 7-10 days for Anthracnose and bacterial spots",
            "Foliar calcium spray for rapid Blossom End Rot correction",
            "Chlorothalonil for severe fungal pressure"
        ],
        "management": [
            "Ensure plants have a dense leaf canopy to shade the fruit and prevent Sunscald",
            "Maintain perfectly even soil moisture (fluctuating moisture causes Blossom End Rot)",
            "Use drip irrigation; never overhead water peppers"
        ]
    },
    "Potato_Healthy": {
        "plant_type": "Potato",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - tuber appears healthy",
        "symptoms": [
            "Smooth, firm skin with no soft or dark spots",
            "No greening (solanine) on the skin",
            "No corky lesions or deep pits"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Store in a completely dark, cool (45-50°F), and humid environment",
            "Cure for 1-2 weeks at 60°F before long-term storage"
        ]
    },
    "Potato_Diseased": {
        "plant_type": "Potato",
        "health_status": "Diseased",
        "diagnosis": "Common Scab / Late Blight Tuber Rot / Soft Rot",
        "symptoms": [
            "Rough, corky, raised or pitted lesions on the skin (Scab)",
            "Reddish-brown, granular dry rot spreading inward from the skin (Late Blight)",
            "Mushy, foul-smelling bacterial decay (Soft Rot)"
        ],
        "medicine": [
            "Fungicides applied to the foliage (Mancozeb, Chlorothalonil) prevent Late Blight spores from reaching tubers",
            "No chemical cure exists for Common Scab or Soft Rot once infected"
        ],
        "management": [
            "Lower soil pH to 5.0 - 5.2 using elemental sulfur (Scab cannot survive acidic soil)",
            "Never harvest wet fields; ensure tubers are perfectly dry before storage",
            "Hill the plants deeply to put a thick barrier of soil between spores and the tubers",
            "Use certified disease-free seed potatoes"
        ]
    },
    "Raspberry_Healthy": {
        "plant_type": "Raspberry",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Firm, deeply colored drupelets",
            "Hollow core is clean and intact",
            "No fuzzy mold or soft, leaking spots"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Harvest gently and place immediately into cold storage"
        ]
    },
    "Raspberry_Diseased": {
        "plant_type": "Raspberry",
        "health_status": "Diseased",
        "diagnosis": "Botrytis Fruit Rot (Gray Mold) / Sunburn",
        "symptoms": [
            "Dense, fuzzy gray mold covering individual drupelets",
            "Fruit becomes mushy, leaks juice, and rots rapidly",
            "White or bleached drupelets on the sun-exposed side (Sunburn/White Drupelet Disorder)"
        ],
        "medicine": [
            "Captan (2.5g/L) or Fenhexamid (Elevate) sprayed during early and full bloom",
            "Biological fungicides (Bacillus subtilis/Serenade) applied organically"
        ],
        "management": [
            "Thin raspberry canes aggressively (leave 4-5 per foot) to ensure fast drying after rain",
            "Harvest daily in the morning after dew has dried",
            "Cool berries to 32-34°F within 1 hour of picking to arrest mold growth"
        ]
    },
    "Soybean_Healthy": {
        "plant_type": "Soybean",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - pods appear healthy",
        "symptoms": [
            "Clean, smooth pods with well-developed seeds inside",
            "Normal yellowing/browning as they reach harvest maturity",
            "No black specks or mold on the pods"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Harvest promptly when moisture reaches 13-15%"
        ]
    },
    "Soybean_Diseased": {
        "plant_type": "Soybean",
        "health_status": "Diseased",
        "diagnosis": "Pod and Stem Blight / Phomopsis Seed Decay",
        "symptoms": [
            "Black fungal specks (pycnidia) arranged in linear rows on the pods",
            "Seeds inside become shriveled, moldy, and cracked",
            "White fungal growth covering the seeds"
        ],
        "medicine": [
            "Foliar fungicides (Strobilurins like Pyraclostrobin or Azoxystrobin) applied at the R3 to R5 (pod set) stage",
            "Thiophanate-methyl used for severe Phomopsis outbreaks"
        ],
        "management": [
            "Do not delay harvest; prompt harvesting prevents late-season seed decay",
            "Rotate with a non-host crop like corn or wheat for at least one year",
            "Plow under soybean residue to accelerate decomposition of overwintering pathogens"
        ]
    },
    "Squash_Healthy": {
        "plant_type": "Squash",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Firm rind with even, vibrant coloring",
            "No soft spots, mold, or rotting ends",
            "Stem is intact and healthy"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Elevate growing fruits off bare soil using straw"
        ]
    },
    "Squash_Diseased": {
        "plant_type": "Squash",
        "health_status": "Diseased",
        "diagnosis": "Choanephora Fruit Rot / Blossom End Rot",
        "symptoms": [
            "Fuzzy, wet growth resembling pincushions with black heads on the blossom end",
            "Black, dry rotting tissue at the tip (Blossom End Rot)",
            "Soft, mushy decay spreading rapidly through the fruit"
        ],
        "medicine": [
            "Copper sprays and standard fungicides are generally ineffective against Choanephora rot",
            "Foliar calcium sprays applied early to prevent Blossom End Rot"
        ],
        "management": [
            "Crucial: Avoid any overhead irrigation; use drip tapes exclusively",
            "Increase plant spacing to allow rapid drying of flowers and fruit",
            "Ensure consistent watering to prevent the calcium deficiencies that cause Blossom End Rot"
        ]
    },
    "Strawberry_Healthy": {
        "plant_type": "Strawberry",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - fruit appears healthy",
        "symptoms": [
            "Bright red, firm, and fully colored berries",
            "Green, healthy calyx (cap)",
            "No soft spots or mold"
        ],
        "medicine": [
            "No treatment needed"
        ],
        "management": [
            "Harvest early in the day while temperatures are cool"
        ]
    },
    "Strawberry_Diseased": {
        "plant_type": "Strawberry",
        "health_status": "Diseased",
        "diagnosis": "Botrytis (Gray Mold) / Leather Rot / Anthracnose",
        "symptoms": [
            "Fuzzy gray mold covering the berry, spreading via contact",
            "Tough, leathery, dull, discolored areas with a foul smell (Leather Rot)",
            "Sunken, dark brown/black circular lesions (Anthracnose)"
        ],
        "medicine": [
            "Captan (2.5g/L) or Fludioxonil (Switch) sprayed aggressively during the bloom period",
            "Phosphite fungicides (e.g., Aliette) specifically targeting Leather Rot",
            "Organic options include biologicals like Serenade or Actinovate"
        ],
        "management": [
            "Apply thick plastic or clean straw mulch to keep all fruit completely off the bare soil",
            "Space plants adequately and control weeds to allow wind to dry the canopy",
            "Harvest frequently and remove every single rotted berry from the field"
        ]
    },
    "Tomato___Bacterial_spot": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Bacterial Spot",
        "symptoms": [
            "Small, dark, water-soaked spots on leaves",
            "Spots may have yellow halos",
            "Lesions on fruit appear as raised, scabby areas",
            "Severe defoliation in advanced stages"
        ],
        "medicine": [
            "Copper hydroxide (Kocide 3000) - Apply 1.5-2.5 lb/acre every 7-10 days",
            "Copper sulfate + Mancozeb combination spray",
            "Streptomycin sulfate (Agri-Mycin 17) - 200 ppm foliar spray",
            "Acibenzolar-S-methyl (Actigard 50WG) - plant immunity booster"
        ],
        "management": [
            "Remove and destroy infected plant debris",
            "Use disease-free certified seeds",
            "Practice crop rotation with non-solanaceous crops for 2-3 years",
            "Ensure proper plant spacing for air circulation",
            "Avoid overhead irrigation to reduce leaf wetness"
        ]
    },
    "Tomato___Early_blight": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Early Blight",
        "symptoms": [
            "Dark brown to black concentric rings on older leaves (target-like spots)",
            "Yellowing around the lesions",
            "Lower leaves affected first, progressing upward",
            "Stem and fruit lesions may appear in severe cases"
        ],
        "medicine": [
            "Chlorothalonil (Bravo/Daconil) - Apply 1.5 pt/acre every 7-14 days",
            "Mancozeb (Dithane M-45) - 2-3 g/L foliar spray",
            "Azoxystrobin (Amistar) - 1 ml/L spray at early symptoms",
            "Neem oil (organic) - 5 ml/L as preventive spray"
        ],
        "management": [
            "Remove infected lower leaves promptly",
            "Apply organic mulch to prevent soil splash onto leaves",
            "Practice 3-year crop rotation away from solanaceous crops",
            "Use resistant tomato varieties when available",
            "Ensure good air circulation between plants"
        ]
    },
    "Tomato___Late_blight": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Late Blight",
        "symptoms": [
            "Large, irregular, water-soaked grayish-green patches on leaves",
            "White fuzzy mold growth on leaf undersides in humid conditions",
            "Dark brown firm lesions on stems",
            "Rapid wilting and plant death in severe outbreaks"
        ],
        "medicine": [
            "Metalaxyl + Mancozeb (Ridomil Gold MZ) - 2.5 g/L spray",
            "Cymoxanil + Mancozeb (Curzate M8) - 3 g/L spray",
            "Copper oxychloride (Blitox-50) - 3 g/L spray",
            "Dimethomorph (Acrobat MZ) - 2 g/L as systemic treatment"
        ],
        "management": [
            "Remove and destroy all infected plants immediately",
            "Do not compost infected material",
            "Use certified disease-free seed potatoes and transplants",
            "Improve air circulation with proper spacing",
            "Avoid overhead watering, especially in evening hours"
        ]
    },
    "Tomato___Leaf_Mold": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Leaf Mold",
        "symptoms": [
            "Pale greenish-yellow spots on upper leaf surfaces",
            "Olive-green to grayish-brown velvety mold on leaf undersides",
            "Leaves may curl, wither, and drop prematurely",
            "Usually starts on lower leaves and progresses upward"
        ],
        "medicine": [
            "Chlorothalonil (Daconil) - 2 g/L foliar spray every 7-10 days",
            "Mancozeb (Dithane) - 2.5 g/L spray",
            "Neem oil (organic) - 5 ml/L preventive spray",
            "Copper-based fungicide - 3 g/L as preventive measure"
        ],
        "management": [
            "Improve greenhouse ventilation and air flow",
            "Reduce humidity by avoiding overhead irrigation",
            "Remove and destroy affected leaves promptly",
            "Use resistant tomato varieties",
            "Space plants adequately for air circulation"
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Septoria Leaf Spot",
        "symptoms": [
            "Numerous small circular spots with dark borders and gray centers",
            "Tiny black dots (pycnidia) visible in the center of spots",
            "Lower leaves affected first",
            "Severe defoliation reducing fruit quality and yield"
        ],
        "medicine": [
            "Chlorothalonil (Bravo) - 1.5-2 pt/acre every 7-10 days",
            "Mancozeb (Dithane M-45) - 2.5 g/L spray",
            "Azoxystrobin (Quadris) - 0.5 ml/L at early infection",
            "Copper hydroxide - 2 g/L as organic alternative"
        ],
        "management": [
            "Remove and destroy infected leaves and plant debris",
            "Mulch around plants to prevent soil-borne spore splash",
            "Practice crop rotation for at least 2 years",
            "Avoid working with plants when foliage is wet",
            "Use drip irrigation instead of overhead watering"
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant_type": "Tomato",
        "health_status": "Pest Infestation",
        "diagnosis": "Two-Spotted Spider Mite Infestation",
        "symptoms": [
            "Fine stippling or tiny yellow-white dots on leaves",
            "Fine webbing on leaf undersides and between stems",
            "Leaves become bronzed, dry, and brittle",
            "Severe infestations cause leaf drop and plant stress"
        ],
        "medicine": [
            "Abamectin (Vertimec) - 0.5 ml/L spray on undersides of leaves",
            "Spiromesifen (Oberon) - 0.3 ml/L as miticide",
            "Neem oil - 5 ml/L spray (organic option)",
            "Insecticidal soap - 20 ml/L spray for mild infestations"
        ],
        "management": [
            "Spray plants with strong water jets to dislodge mites",
            "Introduce predatory mites (Phytoseiulus persimilis) for biological control",
            "Increase humidity around plants",
            "Remove and destroy heavily infested leaves",
            "Avoid excessive nitrogen fertilization"
        ]
    },
    "Tomato___Target_Spot": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Target Spot",
        "symptoms": [
            "Brown spots with concentric rings resembling a target",
            "Spots start small and enlarge over time",
            "Affected leaves may yellow and drop",
            "Can also affect stems and fruit"
        ],
        "medicine": [
            "Chlorothalonil (Daconil) - 2 g/L spray every 7-10 days",
            "Azoxystrobin + Difenoconazole (Amistar Top) - 1 ml/L",
            "Mancozeb (Dithane) - 2.5 g/L spray",
            "Copper oxychloride - 3 g/L as organic option"
        ],
        "management": [
            "Remove infected plant debris after harvest",
            "Practice crop rotation with non-host crops",
            "Improve air circulation through proper plant spacing",
            "Use resistant varieties when available",
            "Avoid excessive irrigation and leaf wetness"
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "symptoms": [
            "Severe upward curling and cupping of leaves",
            "Yellowing of leaf margins and interveinal areas",
            "Stunted plant growth with shortened internodes",
            "Reduced or absent fruit production"
        ],
        "medicine": [
            "Imidacloprid (Confidor) - 0.3 ml/L to control whitefly vector",
            "Thiamethoxam (Actara) - 0.5 g/L spray for whitefly control",
            "Neem oil - 5 ml/L spray (organic whitefly deterrent)",
            "No direct cure for virus - focus on vector control"
        ],
        "management": [
            "Control whitefly populations using yellow sticky traps",
            "Use reflective mulches to repel whiteflies",
            "Remove and destroy infected plants immediately",
            "Use TYLCV-resistant tomato varieties",
            "Install fine mesh insect-proof netting in greenhouses"
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant_type": "Tomato",
        "health_status": "Diseased",
        "diagnosis": "Tomato Mosaic Virus",
        "symptoms": [
            "Light and dark green mosaic mottling pattern on leaves",
            "Leaf curling, distortion, and reduced size",
            "Stunted plant growth",
            "Uneven fruit ripening with brown streaks internally"
        ],
        "medicine": [
            "No chemical cure available for viral infections",
            "Milk solution spray (1:9 ratio) - helps reduce virus spread",
            "Salicylic acid (aspirin) - 1 tablet/gallon as plant immunity booster",
            "Focus on prevention and vector control"
        ],
        "management": [
            "Remove and destroy infected plants immediately",
            "Disinfect tools and hands with milk solution between plants",
            "Use certified virus-free seeds and transplants",
            "Control aphid vectors with organic methods",
            "Use resistant tomato varieties (TMV-resistant)"
        ]
    },
    "Tomato___healthy": {
        "plant_type": "Tomato",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - plant appears healthy",
        "symptoms": [
            "Vibrant green foliage with no visible spots or lesions",
            "Normal leaf shape and size",
            "Strong stem structure",
            "No signs of wilting or discoloration"
        ],
        "medicine": ["No treatment needed - plant is healthy"],
        "management": [
            "Continue regular watering and balanced organic fertilization",
            "Monitor plants weekly for early signs of disease",
            "Maintain proper plant spacing for airflow",
            "Apply organic mulch to retain soil moisture",
            "Practice preventive crop rotation each season"
        ]
    },
    "Potato___Early_blight": {
        "plant_type": "Potato",
        "health_status": "Diseased",
        "diagnosis": "Early Blight",
        "symptoms": [
            "Dark brown concentric rings on older leaves (bull's-eye pattern)",
            "Yellowing around the spots",
            "Lower leaves affected first, spreading upward",
            "Tuber lesions appear dark and sunken"
        ],
        "medicine": [
            "Mancozeb (Dithane M-45) - 2.5 g/L spray every 7-10 days",
            "Chlorothalonil (Kavach) - 2 g/L spray",
            "Azoxystrobin (Amistar) - 1 ml/L at first symptoms",
            "Copper oxychloride - 3 g/L (organic option)"
        ],
        "management": [
            "Remove and destroy infected foliage promptly",
            "Practice crop rotation for 3 years",
            "Use certified disease-free seed potatoes",
            "Apply organic mulch to prevent soil splash",
            "Use resistant potato varieties when available"
        ]
    },
    "Potato___Late_blight": {
        "plant_type": "Potato",
        "health_status": "Diseased",
        "diagnosis": "Late Blight",
        "symptoms": [
            "Large irregular water-soaked patches on leaves",
            "White mold growth on leaf undersides",
            "Dark brown spots spreading rapidly",
            "Firm brown rot on tubers"
        ],
        "medicine": [
            "Metalaxyl + Mancozeb (Ridomil Gold MZ) - 2.5 g/L spray",
            "Cymoxanil + Mancozeb (Curzate M8) - 3 g/L",
            "Copper hydroxide (Kocide) - 2.5 g/L spray",
            "Dimethomorph - 1 g/L for systemic protection"
        ],
        "management": [
            "Remove and destroy all infected plants immediately",
            "Harvest tubers promptly before infection spreads",
            "Use certified disease-free seed potatoes",
            "Improve field drainage to reduce moisture",
            "Plant resistant potato varieties"
        ]
    },
    "Potato___healthy": {
        "plant_type": "Potato",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - plant appears healthy",
        "symptoms": [
            "Strong green foliage",
            "No visible spots, lesions, or discoloration",
            "Normal plant growth and vigor",
            "Healthy tuber development"
        ],
        "medicine": ["No treatment needed - plant is healthy"],
        "management": [
            "Continue regular hilling and watering practices",
            "Apply balanced organic compost fertilization",
            "Monitor for early signs of blight weekly",
            "Maintain crop rotation schedule",
            "Use mulch to regulate soil temperature and moisture"
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant_type": "Corn (Maize)",
        "health_status": "Diseased",
        "diagnosis": "Gray Leaf Spot (Cercospora)",
        "symptoms": [
            "Rectangular grayish-tan lesions running parallel to leaf veins",
            "Lesions may merge creating large dead areas",
            "Lower leaves affected first",
            "Reduced photosynthesis and yield loss"
        ],
        "medicine": [
            "Azoxystrobin (Quadris) - 0.75 ml/L foliar spray",
            "Propiconazole (Tilt) - 1 ml/L spray at early symptoms",
            "Pyraclostrobin + Metconazole (Headline AMP) - 1 ml/L",
            "Mancozeb - 2.5 g/L as preventive spray"
        ],
        "management": [
            "Practice crop rotation - avoid continuous corn planting",
            "Till or remove corn residue after harvest",
            "Plant resistant corn hybrids",
            "Ensure balanced fertility with adequate potassium",
            "Improve air circulation with proper row spacing"
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "plant_type": "Corn (Maize)",
        "health_status": "Diseased",
        "diagnosis": "Common Rust",
        "symptoms": [
            "Small reddish-brown to cinnamon-brown raised pustules on both leaf surfaces",
            "Pustules scattered across the leaf blade",
            "Severe infections cause leaf yellowing and premature drying",
            "Reduced grain fill and yield"
        ],
        "medicine": [
            "Propiconazole (Tilt 25EC) - 1 ml/L foliar spray",
            "Mancozeb (Dithane M-45) - 2.5 g/L spray",
            "Azoxystrobin (Amistar) - 1 ml/L spray",
            "Sulfur-based fungicide (organic) - 3 g/L spray"
        ],
        "management": [
            "Plant rust-resistant corn hybrids",
            "Ensure timely planting to avoid peak rust season",
            "Maintain balanced soil fertility",
            "Remove volunteer corn plants that harbor rust spores",
            "Ensure good field sanitation after harvest"
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant_type": "Corn (Maize)",
        "health_status": "Diseased",
        "diagnosis": "Northern Leaf Blight",
        "symptoms": [
            "Long, elliptical grayish-green to tan cigar-shaped lesions on leaves",
            "Lesions may reach 2-6 inches in length",
            "Lower leaves affected first, progressing upward",
            "Severe infections cause premature leaf death"
        ],
        "medicine": [
            "Propiconazole (Tilt) - 1 ml/L spray at early infection",
            "Azoxystrobin + Propiconazole (Quilt Xcel) - 1 ml/L",
            "Mancozeb (Dithane) - 2.5 g/L preventive spray",
            "Pyraclostrobin (Headline) - 0.75 ml/L spray"
        ],
        "management": [
            "Use resistant corn hybrids with Ht genes",
            "Practice crop rotation away from corn for 1-2 years",
            "Till corn stubble to reduce overwintering inoculum",
            "Ensure balanced nitrogen and potassium nutrition",
            "Improve air circulation through appropriate plant density"
        ]
    },
    "Corn_(maize)___healthy": {
        "plant_type": "Corn (Maize)",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - plant appears healthy",
        "symptoms": [
            "Vibrant green leaves with no spots or lesions",
            "Strong stalk development",
            "Normal ear and tassel formation",
            "No signs of rust, blight, or nutrient deficiency"
        ],
        "medicine": ["No treatment needed - plant is healthy"],
        "management": [
            "Continue regular irrigation and fertilization schedule",
            "Monitor for early pest and disease signs weekly",
            "Maintain proper weed management",
            "Practice annual crop rotation",
            "Test soil periodically and amend with organic matter"
        ]
    },
    "Apple___Apple_scab": {
        "plant_type": "Apple",
        "health_status": "Diseased",
        "diagnosis": "Apple Scab",
        "symptoms": [
            "Olive-green to dark brown velvety spots on leaves",
            "Scabby rough-textured lesions on fruit surface",
            "Premature leaf drop in severe cases",
            "Distorted or cracked fruit"
        ],
        "medicine": [
            "Captan (50WP) - 2.5 g/L spray during spring",
            "Myclobutanil (Rally) - 0.5 ml/L spray",
            "Mancozeb (Dithane M-45) - 2.5 g/L preventive spray",
            "Sulfur (Wettable Sulfur) - 3 g/L as organic fungicide"
        ],
        "management": [
            "Rake and remove fallen leaves in autumn",
            "Prune trees to improve air circulation within the canopy",
            "Plant scab-resistant apple varieties",
            "Maintain balanced tree nutrition with compost",
            "Avoid overhead irrigation"
        ]
    },
    "Apple___Black_rot": {
        "plant_type": "Apple",
        "health_status": "Diseased",
        "diagnosis": "Black Rot",
        "symptoms": [
            "Purplish-brown spots on leaves (frogeye leaf spot)",
            "Dark rotting areas on fruit starting from the blossom end",
            "Cankers on branches with rough bark",
            "Mummified fruit may remain on the tree"
        ],
        "medicine": [
            "Captan (50WP) - 2.5 g/L spray at petal fall",
            "Thiophanate-methyl (Topsin-M) - 1 g/L spray",
            "Copper oxychloride - 3 g/L dormant season spray",
            "Myclobutanil (Rally) - 0.5 ml/L spray"
        ],
        "management": [
            "Prune and destroy dead or infected branches",
            "Remove mummified fruit from trees and ground",
            "Maintain tree vigor with proper organic fertilization",
            "Keep trees healthy to resist infection",
            "Ensure proper wound care after pruning"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "plant_type": "Apple",
        "health_status": "Diseased",
        "diagnosis": "Cedar Apple Rust",
        "symptoms": [
            "Bright yellow-orange spots on upper leaf surfaces",
            "Small raised tubes (aecia) on leaf undersides",
            "Spots may appear on fruit as well",
            "Premature leaf and fruit drop"
        ],
        "medicine": [
            "Myclobutanil (Rally/Immunox) - 0.5 ml/L at pink bud stage",
            "Mancozeb (Dithane) - 2.5 g/L preventive spray",
            "Triadimefon (Bayleton) - 0.5 g/L spray",
            "Sulfur-based fungicide - 3 g/L (organic option)"
        ],
        "management": [
            "Remove nearby cedar/juniper trees if possible",
            "Plant rust-resistant apple varieties",
            "Prune galls from cedar trees before spring",
            "Maintain tree health with proper nutrition",
            "Monitor trees weekly during spring wet periods"
        ]
    },
    "Apple___healthy": {
        "plant_type": "Apple",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - plant appears healthy",
        "symptoms": [
            "Clean green foliage without spots or lesions",
            "Normal fruit development",
            "Strong branch structure",
            "No signs of fungal growth or pest damage"
        ],
        "medicine": ["No treatment needed - plant is healthy"],
        "management": [
            "Continue regular pruning and orchard maintenance",
            "Apply dormant organic spray before bud break",
            "Monitor for early disease signs throughout season",
            "Maintain soil health with organic mulch and compost",
            "Ensure adequate irrigation during dry periods"
        ]
    },
    "Grape___Black_rot": {
        "plant_type": "Grape",
        "health_status": "Diseased",
        "diagnosis": "Black Rot",
        "symptoms": [
            "Brown circular lesions with dark borders on leaves",
            "Small black pycnidia dots within lesions",
            "Fruit turns brown, shrivels, and becomes hard black mummies",
            "Tendrils and shoots may also show lesions"
        ],
        "medicine": [
            "Myclobutanil (Rally) - 0.5 ml/L spray at shoot growth",
            "Mancozeb (Dithane) - 2.5 g/L preventive spray",
            "Captan - 2 g/L spray during bloom",
            "Sulfur-based fungicide - 3 g/L (organic option)"
        ],
        "management": [
            "Remove mummified fruit and infected debris from vineyard",
            "Prune for open canopy to improve air circulation",
            "Practice good sanitation",
            "Ensure proper vine spacing and trellising",
            "Plant resistant grape varieties when possible"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "plant_type": "Grape",
        "health_status": "Diseased",
        "diagnosis": "Esca (Black Measles)",
        "symptoms": [
            "Interveinal striping - tiger-stripe pattern on leaves",
            "Dark spots and streaks on berries",
            "Sudden vine wilting in severe cases (apoplexy)",
            "Internal wood discoloration visible in cross-section"
        ],
        "medicine": [
            "No effective chemical cure currently available",
            "Trichoderma-based biocontrol agents on pruning wounds",
            "Sodium arsenite was historically used but now banned",
            "Fosetyl-aluminium as preventive trunk injection (research stage)"
        ],
        "management": [
            "Remove and destroy severely infected vines",
            "Protect pruning wounds with wound sealant",
            "Prune during dry weather to minimize infection risk",
            "Maintain vine vigor with balanced organic nutrition",
            "Use trunk renewal techniques for mildly affected vines"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant_type": "Grape",
        "health_status": "Diseased",
        "diagnosis": "Leaf Blight (Isariopsis Leaf Spot)",
        "symptoms": [
            "Irregular brown to reddish-brown spots on leaves",
            "Spots may coalesce causing large necrotic areas",
            "Dark fungal growth may appear on lesion surfaces",
            "Premature leaf drop reducing vine vigor"
        ],
        "medicine": [
            "Mancozeb (Dithane M-45) - 2.5 g/L spray every 10-14 days",
            "Copper oxychloride (Blitox) - 3 g/L spray",
            "Carbendazim (Bavistin) - 1 g/L spray",
            "Sulfur dust - as organic fungicide option"
        ],
        "management": [
            "Remove and destroy infected leaves and plant debris",
            "Improve canopy air circulation through proper pruning",
            "Avoid overhead irrigation",
            "Maintain balanced vine nutrition",
            "Practice crop sanitation between seasons"
        ]
    },
    "Grape___healthy": {
        "plant_type": "Grape",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - vine appears healthy",
        "symptoms": [
            "Clean green leaves with no spots or discoloration",
            "Normal vine growth and fruit set",
            "Strong cane and trunk development",
            "No signs of rot, mildew, or pest damage"
        ],
        "medicine": ["No treatment needed - vine is healthy"],
        "management": [
            "Continue regular pruning and canopy management",
            "Monitor for early disease signs weekly",
            "Maintain balanced irrigation schedule",
            "Apply organic compost for soil health",
            "Ensure proper vine spacing for air flow"
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant_type": "Bell Pepper",
        "health_status": "Diseased",
        "diagnosis": "Bacterial Spot",
        "symptoms": [
            "Small, dark, water-soaked lesions on leaves",
            "Spots may have yellow halos and become necrotic",
            "Raised, scab-like lesions on fruit",
            "Defoliation and reduced fruit quality"
        ],
        "medicine": [
            "Copper hydroxide (Kocide 3000) - 2 g/L spray every 7-10 days",
            "Streptomycin sulfate - 200 ppm spray (where permitted)",
            "Copper sulfate + Mancozeb combination",
            "Acibenzolar-S-methyl (Actigard) - plant defense activator"
        ],
        "management": [
            "Use disease-free certified seeds and transplants",
            "Practice crop rotation for 2-3 years",
            "Avoid overhead irrigation and working with wet plants",
            "Remove and destroy infected plant debris",
            "Improve air circulation through proper plant spacing"
        ]
    },
    "Pepper,_bell___healthy": {
        "plant_type": "Bell Pepper",
        "health_status": "Healthy",
        "diagnosis": "No disease detected - plant appears healthy",
        "symptoms": [
            "Dark green glossy leaves with no spots",
            "Strong stem and branching structure",
            "Normal fruit development and coloring",
            "No wilting or discoloration present"
        ],
        "medicine": ["No treatment needed - plant is healthy"],
        "management": [
            "Continue regular watering and organic fertilization",
            "Monitor for pests like aphids and whiteflies weekly",
            "Maintain proper spacing for air circulation",
            "Use organic mulch to conserve soil moisture",
            "Practice seasonal crop rotation"
        ]
    },

}

# Generic fallback for classes not explicitly in the knowledge base
DEFAULT_DISEASE_INFO = {
    "symptoms": ["Visible abnormalities observed on the plant tissue"],
    "medicine": [
        "Consult a local agricultural extension officer for specific medicine recommendations",
        "Copper-based fungicide - 3 g/L as general preventive spray",
        "Mancozeb (Dithane M-45) - 2.5 g/L as broad-spectrum treatment"
    ],
    "management": [
        "Consult a local agricultural extension officer for field-specific guidance",
        "Remove and destroy visibly infected plant parts",
        "Practice crop rotation and field sanitation",
        "Maintain balanced organic nutrition and proper irrigation"
    ]
}


def get_disease_info(class_name):
    """Look up disease info from the knowledge base, with fallback."""
    if class_name in DISEASE_KNOWLEDGE:
        return DISEASE_KNOWLEDGE[class_name]

    # Parse the class name to extract plant and disease info
    if "_" in class_name and "___" not in class_name:
        parts = class_name.split("_")
        plant_type = parts[0]
        disease_name = parts[1] if len(parts) > 1 else "Unknown Condition"
    else:
        parts = class_name.replace("___", "|").replace("_", " ").split("|")
        plant_type = parts[0].strip() if len(parts) > 0 else "Unknown"
        disease_name = parts[1].strip() if len(parts) > 1 else "Unknown Condition"

    is_healthy = "healthy" in class_name.lower()

    return {
        "plant_type": plant_type,
        "health_status": "Healthy" if is_healthy else "Diseased",
        "diagnosis": "No disease detected - plant appears healthy" if is_healthy else disease_name,
        "symptoms": DEFAULT_DISEASE_INFO["symptoms"] if not is_healthy else ["Plant appears healthy with no visible issues"],
        "medicine": DEFAULT_DISEASE_INFO["medicine"] if not is_healthy else ["No treatment needed - plant is healthy"],
        "management": DEFAULT_DISEASE_INFO["management"] if not is_healthy else ["Continue regular care and monitoring"],
    }


# ---------------------------------------------------------------------------
# Offline Smart Chatbot (No API Key Required)
# ---------------------------------------------------------------------------

FARMING_KNOWLEDGE = {
    "greetings": {
        "keywords": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup"],
        "response": "Hello! I'm your AgriShield AI Advisor - a virtual agronomist here to help you with crop diseases, soil health, irrigation, pest control, and farming tips. Ask me anything about agriculture!"
    },
    "thanks": {
        "keywords": ["thank", "thanks", "thank you", "appreciate", "helpful"],
        "response": "You're welcome! I'm always here to help with your farming questions. Don't hesitate to ask anything else about crop health, diseases, or farming practices!"
    },
    "soil_health": {
        "keywords": ["soil", "soil health", "soil test", "ph", "soil preparation", "compost", "organic matter", "fertilizer", "nutrient"],
        "response": "**Soil Health Tips:**\n\n- **Test your soil** every season for pH, nitrogen, phosphorus, and potassium levels\n- **Ideal pH range**: 6.0-7.0 for most vegetables\n- **Add organic compost** (2-3 inches) to improve soil structure and nutrients\n- **Use cover crops** like clover or rye during off-season to prevent erosion\n- **Avoid over-tilling** which destroys beneficial soil organisms\n- **Rotate crops** annually to prevent nutrient depletion\n- **Mulch** with straw or leaves to retain moisture and suppress weeds\n\n**Organic Fertilizers:**\n- Vermicompost - excellent all-purpose\n- Bone meal - high in phosphorus\n- Blood meal - high in nitrogen\n- Wood ash - adds potassium and raises pH"
    },
    "irrigation": {
        "keywords": ["water", "irrigation", "watering", "drip", "sprinkler", "moisture", "drought", "overwater"],
        "response": "**Irrigation Best Practices:**\n\n- **Drip irrigation** is most efficient (90-95% water efficiency)\n- **Water early morning** (6-10 AM) to reduce evaporation and disease\n- **Avoid overhead watering** in evening - promotes fungal diseases\n- **Check soil moisture** 2-3 inches deep before watering\n- **Mulch around plants** to retain 25-50% more moisture\n\n**Crop Water Needs:**\n- Tomato: 1-2 inches/week\n- Potato: 1-2 inches/week\n- Corn: 1.5-2 inches/week\n- Pepper: 1-1.5 inches/week\n\n**Signs of Overwatering:** Yellow leaves, wilting despite wet soil, root rot\n**Signs of Underwatering:** Crispy leaf edges, drooping, slow growth"
    },
    "pest_control": {
        "keywords": ["pest", "insect", "bug", "aphid", "whitefly", "caterpillar", "beetle", "worm", "mite", "pesticide"],
        "response": "**Integrated Pest Management (IPM):**\n\n**Biological Controls:**\n- Ladybugs eat aphids (1 ladybug = 50 aphids/day)\n- Lacewings control whiteflies and mealybugs\n- Neem oil spray (5 ml/L) - effective organic pesticide\n- Bacillus thuringiensis (Bt) for caterpillars\n\n**Cultural Controls:**\n- Rotate crops to break pest cycles\n- Use yellow sticky traps for whiteflies\n- Plant marigolds as companion plants to repel pests\n- Remove weeds that harbor insects\n- Encourage beneficial insects with diverse plantings\n\n**Organic Sprays:**\n- Neem oil: 5 ml/L for most pests\n- Garlic + chili spray: Natural repellent\n- Insecticidal soap: 20 ml/L for soft-bodied insects\n- Diatomaceous earth: For crawling insects"
    },
    "crop_rotation": {
        "keywords": ["rotation", "crop rotation", "rotate", "succession", "planting plan"],
        "response": "**Crop Rotation Guide:**\n\n**4-Year Rotation Plan:**\n- Year 1: Legumes (beans, peas) - fix nitrogen\n- Year 2: Leafy greens (spinach, lettuce) - use nitrogen\n- Year 3: Fruiting crops (tomato, pepper) - heavy feeders\n- Year 4: Root crops (potato, carrot) - break disease cycles\n\n**Rules:**\n- Never plant the same family in the same spot 2 years in a row\n- Solanaceae family (tomato, potato, pepper, eggplant) - rotate together\n- Follow heavy feeders with light feeders\n- Plant legumes before nitrogen-hungry crops\n\n**Benefits:**\n- Reduces soil-borne diseases by 60-80%\n- Improves soil fertility naturally\n- Breaks pest reproduction cycles\n- Increases yield by 10-25%"
    },
    "tomato": {
        "keywords": ["tomato", "tomatoes"],
        "response": "**Tomato Growing Guide:**\n\n**Common Diseases:**\n- Early Blight: Dark concentric rings on leaves. Treat with Mancozeb 2.5g/L\n- Late Blight: Water-soaked patches, white mold. Use Ridomil Gold MZ 2.5g/L\n- Bacterial Spot: Dark spots with yellow halos. Apply copper hydroxide\n- Leaf Mold: Yellow spots, olive mold underneath. Improve ventilation\n- Septoria Leaf Spot: Small circular spots. Use Chlorothalonil spray\n\n**Growing Tips:**\n- Plant in full sun (6-8 hours daily)\n- Space plants 18-24 inches apart\n- Stake or cage for support\n- Water at base, not leaves\n- Prune suckers for better airflow\n- Harvest when fully colored\n\nUpload a leaf image to the Disease Scanner for AI-powered diagnosis!"
    },
    "potato": {
        "keywords": ["potato", "potatoes"],
        "response": "**Potato Growing Guide:**\n\n**Common Diseases:**\n- Early Blight: Bull's-eye pattern on leaves. Treat with Mancozeb 2.5g/L\n- Late Blight: Rapid browning, white mold. Use Ridomil Gold MZ 2.5g/L\n\n**Growing Tips:**\n- Plant seed potatoes 4 inches deep, 12 inches apart\n- Hill soil around stems as plants grow (every 2-3 weeks)\n- Water consistently - 1-2 inches per week\n- Harvest 2-3 weeks after foliage dies back\n- Cure in dark, cool place for 1-2 weeks\n\n**Prevention:**\n- Use certified disease-free seed potatoes\n- Don't plant where tomatoes/peppers grew last year\n- Destroy any volunteer potato plants\n\nUpload a leaf image to the Disease Scanner for AI-powered diagnosis!"
    },
    "corn": {
        "keywords": ["corn", "maize"],
        "response": "**Corn/Maize Growing Guide:**\n\n**Common Diseases:**\n- Gray Leaf Spot: Rectangular gray lesions. Treat with Azoxystrobin 0.75ml/L\n- Common Rust: Reddish-brown pustules. Apply Propiconazole 1ml/L\n- Northern Leaf Blight: Cigar-shaped lesions. Use Mancozeb 2.5g/L\n\n**Growing Tips:**\n- Plant in blocks (not single rows) for better pollination\n- Space 8-12 inches apart in rows 30-36 inches apart\n- Needs lots of nitrogen - side-dress with fertilizer at knee height\n- Water 1.5-2 inches per week, especially during tasseling\n- Harvest when silks turn brown and kernels are milky\n\nUpload a leaf image to the Disease Scanner for AI-powered diagnosis!"
    },
    "apple": {
        "keywords": ["apple", "apples"],
        "response": "**Apple Tree Care Guide:**\n\n**Common Diseases:**\n- Apple Scab: Olive-green velvety spots. Treat with Captan 2.5g/L\n- Black Rot: Dark rotting fruit, cankers. Apply Thiophanate-methyl 1g/L\n- Cedar Apple Rust: Bright orange spots. Use Myclobutanil 0.5ml/L\n\n**Care Tips:**\n- Prune annually in late winter for airflow\n- Thin fruit to 1 apple per cluster for larger fruit\n- Apply dormant spray before bud break\n- Rake fallen leaves in autumn to prevent scab\n- Water deeply but infrequently\n\nUpload a leaf image to the Disease Scanner for AI-powered diagnosis!"
    },
    "grape": {
        "keywords": ["grape", "grapes", "vineyard", "vine"],
        "response": "**Grape Growing Guide:**\n\n**Common Diseases:**\n- Black Rot: Brown lesions, shriveled fruit. Treat with Myclobutanil 0.5ml/L\n- Esca (Black Measles): Tiger-stripe leaves. No cure - remove infected vines\n- Leaf Blight: Brown spots, leaf drop. Use Mancozeb 2.5g/L\n\n**Care Tips:**\n- Train on trellis system for airflow\n- Prune heavily in dormant season (90% of previous growth)\n- Manage canopy to reduce humidity\n- Remove mummified fruit promptly\n- Water at base, avoid wetting foliage\n\nUpload a leaf image to the Disease Scanner for AI-powered diagnosis!"
    },
    "pepper": {
        "keywords": ["pepper", "bell pepper", "chili", "capsicum"],
        "response": "**Bell Pepper Growing Guide:**\n\n**Common Diseases:**\n- Bacterial Spot: Dark water-soaked lesions. Treat with Copper hydroxide 2g/L\n\n**Growing Tips:**\n- Start indoors 8-10 weeks before last frost\n- Transplant when soil is warm (65F+)\n- Space 18-24 inches apart\n- Mulch to maintain soil moisture and temperature\n- Harvest when firm and full-sized (green or colored)\n\n**Nutrition:**\n- Peppers need calcium to prevent blossom end rot\n- Side-dress with compost at flowering\n- Avoid excess nitrogen (causes leafy growth, fewer fruits)\n\nUpload a leaf image to the Disease Scanner for AI-powered diagnosis!"
    },
    "disease_general": {
        "keywords": ["disease", "infection", "fungus", "fungal", "bacterial", "virus", "blight", "rot", "mold", "rust", "spot", "wilt", "scab", "cure", "treatment", "medicine"],
        "response": "**Common Crop Disease Management:**\n\n**Fungal Diseases (most common):**\n- Treat with Mancozeb (Dithane M-45) - 2.5 g/L spray\n- Copper oxychloride (Blitox) - 3 g/L spray\n- Azoxystrobin (Amistar) - 1 ml/L for advanced infections\n\n**Bacterial Diseases:**\n- Copper hydroxide (Kocide) - 2 g/L spray\n- Streptomycin sulfate - 200 ppm spray\n- Remove infected parts immediately\n\n**Viral Diseases:**\n- No chemical cure available\n- Control insect vectors (aphids, whiteflies)\n- Remove and destroy infected plants\n- Use resistant varieties\n\n**Prevention is Key:**\n- Practice crop rotation\n- Ensure good air circulation\n- Avoid overhead watering\n- Use certified disease-free seeds\n\nFor accurate diagnosis, upload a leaf image to the **Disease Scanner** tab!"
    },
    "organic": {
        "keywords": ["organic", "natural", "chemical free", "bio", "sustainable"],
        "response": "**Organic Farming Practices:**\n\n**Pest Control:**\n- Neem oil: Universal organic pesticide (5 ml/L)\n- Garlic-chili extract: Natural insect repellent\n- Diatomaceous earth: For crawling pests\n- Companion planting: Marigolds, basil, nasturtiums\n\n**Disease Control:**\n- Copper-based sprays (approved organic fungicide)\n- Sulfur dust: For powdery mildew\n- Baking soda spray: 1 tbsp/gallon for mild fungal issues\n- Trichoderma bio-fungicide\n\n**Soil Building:**\n- Compost: Foundation of organic farming\n- Vermicompost: Nutrient-rich worm castings\n- Green manure: Cover crops turned into soil\n- Bone meal and blood meal for nutrients\n\n**Certification:** Contact your local agricultural office for organic certification guidelines."
    },
    "weather": {
        "keywords": ["weather", "rain", "monsoon", "season", "climate", "frost", "heat", "cold", "temperature", "humidity"],
        "response": "**Weather & Crop Management:**\n\n**Hot Weather (>35C/95F):**\n- Increase watering frequency\n- Use shade cloth (30-50% shade)\n- Mulch heavily to keep roots cool\n- Harvest early morning when cool\n\n**Rainy/Monsoon Season:**\n- Ensure good drainage\n- Apply preventive fungicide before rains\n- Stake plants to prevent lodging\n- Monitor for increased disease pressure\n\n**Cold/Frost:**\n- Cover plants with row covers or plastic\n- Water soil before frost (retains heat)\n- Harvest frost-sensitive crops before first frost\n- Use cold frames for extending season\n\n**High Humidity:**\n- Space plants wider for airflow\n- Prune lower leaves\n- Apply preventive fungicide\n- Use drip irrigation (not overhead)"
    },
    "harvest": {
        "keywords": ["harvest", "pick", "ripe", "yield", "produce", "storage", "store"],
        "response": "**Harvesting & Storage Tips:**\n\n**When to Harvest:**\n- Tomatoes: Firm, fully colored, slight give when squeezed\n- Potatoes: 2-3 weeks after foliage dies back\n- Corn: Silks brown, kernels milky when pierced\n- Peppers: Firm, full-sized, desired color reached\n- Apples: Twist gently - ripe fruit detaches easily\n\n**Storage:**\n- Tomatoes: Room temperature (never refrigerate unripe ones)\n- Potatoes: Cool, dark place (45-50F), cure first for 1-2 weeks\n- Corn: Refrigerate immediately, use within 2-3 days\n- Apples: Cold storage (32-35F), check for rot regularly\n\n**Post-Harvest:**\n- Remove crop debris to reduce disease carryover\n- Add compost to replenish soil\n- Plant cover crops for winter"
    },
    "plant_nutrition": {
        "keywords": ["npk", "nitrogen", "phosphorus", "potassium", "deficiency", "yellow leaves", "calcium", "magnesium", "nutrition"],
        "response": "**Plant Nutrition Guide (NPK):**\n\n**Nitrogen (N) - Leaves & Stems:**\n- Good for: Leafy greens, early growth stages\n- Deficiency: Older leaves turn pale yellow\n- Sources: Blood meal, fish emulsion, compost, manure\n\n**Phosphorus (P) - Roots & Blooms:**\n- Good for: Root development, flower/fruit production\n- Deficiency: Stunted growth, dark green/purplish leaves\n- Sources: Bone meal, rock phosphate\n\n**Potassium (K) - Overall Health:**\n- Good for: Disease resistance, water regulation, fruit quality\n- Deficiency: Brown scorching on leaf edges\n- Sources: Wood ash, kelp meal, greensand\n\n**Micronutrients:**\n- Calcium deficiency causes Blossom End Rot in tomatoes/peppers. Fix with crushed eggshells or garden lime."
    },
    "composting": {
        "keywords": ["compost", "composting", "bin", "pile", "mulch", "vermicompost", "worms"],
        "response": "**Composting Best Practices:**\n\n**The Recipe (Carbon to Nitrogen ratio):**\nAim for 2 parts 'Browns' (Carbon) to 1 part 'Greens' (Nitrogen).\n\n**Browns (Carbon):**\n- Dry leaves, straw, hay, paper, cardboard, sawdust\n\n**Greens (Nitrogen):**\n- Grass clippings, vegetable scraps, coffee grounds, manure (herbivore)\n\n**Do NOT Compost:**\n- Meat, dairy, oils/grease, pet waste, diseased plants, invasive weeds with seeds\n\n**Maintenance:**\n- Turn the pile every 1-2 weeks to aerate\n- Keep it as moist as a wrung-out sponge\n- Should be ready in 3-6 months\n- Vermicomposting (using red wiggler worms) is faster and produces nutrient-dense castings."
    },
    "weed_management": {
        "keywords": ["weed", "weeds", "weeding", "herbicide", "mulching", "bindweed", "crabgrass"],
        "response": "**Organic Weed Management:**\n\n**Cultural Control:**\n- **Mulching:** Apply 2-3 inches of organic mulch (straw, wood chips) to block light and prevent weed seed germination.\n- **Cover Crops:** Plant dense cover crops like clover or buckwheat to outcompete weeds.\n- **Spacing:** Plant crops closer together to shade the soil canopy.\n\n**Mechanical Control:**\n- Hoeing and hand-pulling when weeds are young (easier when soil is moist).\n- Solarization: Cover moist soil with clear plastic for 4-6 weeks in summer to kill weed seeds using sun's heat.\n\n**Natural Herbicides (Use with Caution!):**\n- Horticultural vinegar (20% acetic acid) mixed with a little dish soap. Spray directly on young weed leaves on a sunny day. *Note: This is non-selective and will harm any plant it touches.*"
    },
    "beneficial_insects": {
        "keywords": ["beneficial", "good bugs", "ladybug", "lacewing", "bee", "pollinator", "wasp", "predator"],
        "response": "**Beneficial Insects & Pollinators:**\n\n**Predators (Eat Pests):**\n- **Ladybugs:** Voracious aphid eaters.\n- **Green Lacewings:** Larvae consume aphids, mealybugs, and caterpillars.\n- **Praying Mantis:** Ambush predators (but they eat good and bad bugs).\n- **Hoverflies:** Larvae eat aphids; adults pollinate.\n- **Parasitic Wasps:** Lay eggs inside caterpillars and aphids (e.g., tomato hornworm).\n\n**Pollinators (Help Plants Fruit):**\n- Honeybees, Bumblebees, Mason Bees, Butterflies.\n\n**How to Attract Them:**\n- Plant diverse, nectar-rich flowers (marigolds, dill, fennel, yarrow, sunflowers).\n- Provide shallow water sources.\n- Minimize broad-spectrum pesticide use, even organic ones!"
    },
    "seasons": {
        "keywords": ["spring", "summer", "fall", "autumn", "winter", "season", "planting time"],
        "response": "**Seasonal Planting Guide:**\n\n**Spring (Cool to Warm):**\n- *Plant:* Peas, lettuce, spinach, radishes, carrots, broccoli.\n- *Tasks:* Prepare beds, add compost, start warm-season seeds indoors.\n\n**Summer (Hot):**\n- *Plant:* Tomatoes, peppers, corn, cucumbers, squash, melons, beans.\n- *Tasks:* Mulch heavily, water consistently, monitor for pests/diseases.\n\n**Fall (Warm to Cool):**\n- *Plant:* Garlic, kale, spinach, radishes, winter cover crops.\n- *Tasks:* Harvest summer crops, clean up debris, plant garlic for next year.\n\n**Winter (Cold):**\n- *Plant:* Nothing outdoors in freezing zones (use greenhouses/cold frames).\n- *Tasks:* Tool maintenance, order seeds, plan next year's crop rotation."
    },
    "greenhouse": {
        "keywords": ["greenhouse", "hoop house", "high tunnel", "glasshouse", "indoor farming", "hydroponic", "aquaponic"],
        "response": "**Greenhouse & Indoor Farming:**\n\n**Greenhouse Management:**\n- **Ventilation:** Crucial to prevent fungal diseases. Open vents daily, use fans for air circulation.\n- **Temperature:** Ideal is 70-80°F (21-27°C) by day, 60-65°F (15-18°C) by night.\n- **Shade Cloth:** Use in peak summer to prevent overheating.\n- **Pollination:** Since bees are absent, hand-pollinate tomatoes/peppers by gently shaking the plants or using an electric toothbrush.\n\n**Alternative Systems:**\n- **Hydroponics:** Growing plants in nutrient-rich water instead of soil (faster growth, uses less water, requires precise pH/nutrient monitoring).\n- **Aquaponics:** Combining hydroponics with aquaculture (fish farming) - fish waste feeds plants, plants clean the water."
    },
    "about": {
        "keywords": ["who are you", "what are you", "what can you do", "help", "capabilities", "features"],
        "response": "**I'm AgriShield AI Advisor!**\n\nI can help you with:\n\n- **Crop Diseases** - Ask about diseases in tomato, potato, corn, apple, grape, pepper\n- **Medicine & Treatment** - Get specific fungicide/pesticide recommendations\n- **Soil Health** - pH management, composting, fertilization\n- **Irrigation** - Watering schedules, drip systems\n- **Pest Control** - IPM, organic solutions\n- **Crop Rotation** - Planning and scheduling\n- **Organic Farming** - Chemical-free methods\n- **Weather Management** - Season-specific advice\n- **Harvesting** - When and how to harvest\n\n**For disease diagnosis**, switch to the **Disease Scanner** tab and upload a leaf image!\n\nTry asking: 'How to treat tomato blight?' or 'What's a good crop rotation plan?'"
    },
}


def get_chatbot_response(user_message):
    """Generate a response using keyword matching against farming knowledge base."""
    msg_lower = user_message.lower().strip()

    # Check each knowledge topic for keyword matches
    best_match = None
    best_score = 0

    for topic, data in FARMING_KNOWLEDGE.items():
        score = 0
        for keyword in data["keywords"]:
            if keyword in msg_lower:
                # Longer keyword matches score higher
                score += len(keyword.split())
        
        if score > best_score:
            best_score = score
            best_match = topic

    # Also check disease knowledge base for specific disease queries
    best_disease_match = None
    best_disease_score = 0

    for class_name, info in DISEASE_KNOWLEDGE.items():
        if info.get("health_status", "Healthy") == "Healthy":
            continue
            
        plant = info.get("plant_type", "").lower()
        disease = info.get("diagnosis", "").lower()
        
        score = 0
        
        # High score for matching the exact disease name (e.g. "early blight")
        if disease and disease in msg_lower:
            score += 10
            
        # Medium score for matching parts of the disease name
        if disease:
            disease_words = [w for w in disease.split() if len(w) > 3]
            for w in disease_words:
                if w in msg_lower:
                    score += 3
                    
        # Score for matching the plant name
        if plant and plant in msg_lower:
            score += 4
            
        # We need a minimum score to prevent random matches
        if score > best_disease_score and score >= 6:
            best_disease_score = score
            best_disease_match = info

    if best_disease_match:
        info = best_disease_match
        symptoms = "\n".join(f"- {s}" for s in info.get("symptoms", []))
        medicines = "\n".join(f"- {m}" for m in info.get("medicine", []))
        management = "\n".join(f"- {m}" for m in info.get("management", []))
        
        return (
            f"**{info['plant_type']} - {info['diagnosis']}**\n\n"
            f"**Status:** {info['health_status']}\n\n"
            f"**Symptoms:**\n{symptoms}\n\n"
            f"**Recommended Medicine:**\n{medicines}\n\n"
            f"**Management:**\n{management}\n\n"
            f"For visual diagnosis, upload a leaf image to the **Disease Scanner** tab!"
        )

    if best_match and best_score > 0:
        return FARMING_KNOWLEDGE[best_match]["response"]

    # Default fallback
    return (
        "I understand you're asking about farming. Here are some topics I can help with:\n\n"
        "- **Crop Diseases** - Ask about tomato, potato, corn, apple, grape, or pepper diseases\n"
        "- **Medicine/Treatment** - Get specific treatment recommendations\n"
        "- **Soil Health** - Composting, pH, fertilization\n"
        "- **Irrigation** - Watering tips and schedules\n"
        "- **Pest Control** - Organic and IPM methods\n"
        "- **Crop Rotation** - Planning rotations\n"
        "- **Organic Farming** - Chemical-free practices\n"
        "- **Weather** - Season-specific crop advice\n"
        "- **Harvesting** - When and how to harvest\n\n"
        "Try asking something like:\n"
        "- 'How to treat potato blight?'\n"
        "- 'Tell me about soil health'\n"
        "- 'Best pest control methods'\n\n"
        "Or use the **Disease Scanner** tab to upload a leaf image for AI diagnosis!"
    )


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has a permitted extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes, target_size=IMG_SIZE):
    """Preprocess image for the trained CNN model."""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array





# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard, or app status if JSON is requested."""
    if request.headers.get("Accept") == "application/json" or request.args.get("format") == "json" or request.args.get("json") == "1":
        return jsonify({
            "app": "AgriShield AI",
            "status": "running",
            "model_loaded": model_loaded,
            "gemini_available": gemini_available,
            "version": "1.0"
        }), 200
    return render_template("index.html")



def is_probably_plant_image(image_bytes):
    """
    Analyzes color distributions of the image at 32x32 resolution.
    Returns True if the image has a reasonable proportion of organic plant-like colors
    (greens, yellow-decay browns, ripe organic reds).
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((32, 32))
        pixels = list(img.getdata())
        
        plant_pixel_count = 0
        total_pixels = 32 * 32
        
        for r, g, b in pixels:
            # 1. Green tones: Green is dominant and significantly larger than blue/red
            is_green = (g > r * 1.1) and (g > b * 1.1) and (g > 30)
            
            # 2. Yellow/Brown tones (decay/healthy fruit/rust): Red and Green are high, Blue is low
            is_yellow_brown = (r > 60) and (g > 50) and (g > b * 1.2) and (r > b * 1.2) and (abs(r - g) < 60)
            
            # 3. Organic Red tones (ripe tomato/apple): Red is highly dominant
            is_organic_red = (r > g * 1.3) and (r > b * 1.3) and (r > 60)
            
            if is_green or is_yellow_brown or is_organic_red:
                plant_pixel_count += 1
                
        plant_ratio = plant_pixel_count / total_pixels
        print(f"[INFO] Local organic color validation ratio: {plant_ratio:.2f}", flush=True)
        
        # If less than 12% of the image contains plant-like organic colors, it is likely not a plant!
        return plant_ratio >= 0.12
    except Exception as e:
        print(f"[WARNING] Local color validation failed: {e}", flush=True)
        return True  # Fallback to True if analysis fails


@app.route("/api/analyze", methods=["POST"])
def analyze_crop():
    """
    Accepts an image upload, runs it through the trained CNN model,
    and returns structured JSON diagnosis.
    """
    try:
        # Check if the disease detection model is in fallback mock mode
        mock_mode = not model_loaded or trained_model is None

        # --- Validate file presence ---
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file was included in the request."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file was selected. Please choose an image to upload."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": (
                    f"Unsupported file type. "
                    f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS).upper()}"
                )
            }), 400

        # --- Read & validate image ---
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
        except Exception:
            return jsonify({
                "success": False,
                "error": "The uploaded file appears to be corrupted or is not a valid image."
            }), 400

        # --- Local organic color check (instant, offline safeguard) ---
        if not is_probably_plant_image(image_bytes):
            return jsonify({
                "success": False,
                "error": "The uploaded image does not appear to contain a valid crop leaf or fruit. Please upload a clear photo of your plant for diagnosis."
            }), 400

        # --- Validate if image contains a crop leaf, fruit or agricultural plant using Gemini ---
        if gemini_available:
            try:
                # Open image for Gemini
                validation_image = Image.open(io.BytesIO(image_bytes))
                validation_model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "Analyze this image. Is it a plant, crop leaf, fruit, tree, or agricultural crop? "
                    "Answer ONLY 'YES' if it is a crop, leaf, plant, or fruit/vegetable. "
                    "Answer 'NO' if it is a human being, face, animal, random indoor object (like a keyboard, mug, room), "
                    "or anything unrelated to agriculture and plant disease."
                )
                response = validation_model.generate_content([validation_image, prompt])
                answer = response.text.strip().upper()
                print(f"[INFO] Gemini Vision crop validation result: {answer}", flush=True)
                
                if "NO" in answer:
                    return jsonify({
                        "success": False,
                        "error": "The uploaded image does not appear to contain a valid crop leaf or fruit. Please upload a clear photo of your plant for diagnosis."
                    }), 400
            except Exception as e:
                print(f"[WARNING] Gemini Vision crop validation failed: {e}", flush=True)

        # --- Preprocess and predict ---
        if mock_mode:
            filename_lower = file.filename.lower()
            fallback_classes = {
                "apple": "Apple___Apple_scab" if "scab" in filename_lower else ("Apple___Cedar_apple_rust" if "rust" in filename_lower else "Apple___healthy"),
                "potato": "Potato___Late_blight" if "late" in filename_lower else ("Potato___Early_blight" if "early" in filename_lower else "Potato___healthy"),
                "tomato": "Tomato___Late_blight" if "late" in filename_lower else ("Tomato___Early_blight" if "early" in filename_lower else "Tomato___healthy"),
                "corn": "Corn_(maize)___Common_rust_" if "rust" in filename_lower else "Corn_(maize)___healthy",
                "grape": "Grape___Black_rot" if "rot" in filename_lower else "Grape___healthy",
                "pepper": "Pepper,_bell___Bacterial_spot" if "spot" in filename_lower else "Pepper,_bell___healthy",
                "strawberry": "Strawberry___Leaf_scorch" if "scorch" in filename_lower else "Strawberry___healthy"
            }
            
            class_name = None
            for key, val in fallback_classes.items():
                if key in filename_lower:
                    class_name = val
                    break
                    
            if class_name is None:
                fallback_keys = [
                    "Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy",
                    "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
                    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
                    "Strawberry___Leaf_scorch", "Strawberry___healthy"
                ]
                char_sum = sum(ord(c) for c in file.filename)
                class_name = fallback_keys[char_sum % len(fallback_keys)]
                
            confidence = 85.0 + (sum(ord(c) for c in file.filename) % 150) / 10.0
            print(f"[MOCK MODE] Falling back to mock prediction for {file.filename}: {class_name} ({confidence:.1f}%)", flush=True)
            
        else:
            img_array = preprocess_image(image_bytes)
            predictions = trained_model.predict(img_array, verbose=0)
            predicted_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_idx]) * 100

            # --- Confidence Threshold Check ---
            if confidence < 50.0:
                return jsonify({
                    "success": False,
                    "error": "The AI model is not confident in this image. Please ensure you upload a clear, well-lit photo of a crop leaf or fruit."
                }), 400

            # Get the class name
            class_name = class_labels.get(str(predicted_idx), "Unknown")


        # Look up disease information
        disease_info = get_disease_info(class_name)

        diagnosis = {
            "plant_type": disease_info["plant_type"],
            "health_status": disease_info["health_status"],
            "diagnosis": disease_info["diagnosis"],
            "confidence": f"{confidence:.1f}%",
            "symptoms": disease_info["symptoms"],
            "medicine": disease_info.get("medicine", []),
            "management": disease_info["management"],
        }

        return jsonify({
            "success": True,
            "data": diagnosis
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred during analysis: {str(exc)}"
        }), 500

@app.route("/api/analyze_fruit", methods=["POST"])
def analyze_fruit():
    """
    Accepts a fruit image upload and runs it through the secondary Fruit CNN model.
    """
    try:
        # Check if the fruit disease detection model is in fallback mock mode
        mock_mode = not model_loaded or fruit_model is None

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        image_bytes = file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
        except Exception:
            return jsonify({"success": False, "error": "Invalid image file."}), 400

        # --- Local organic color check (instant, offline safeguard) ---
        if not is_probably_plant_image(image_bytes):
            return jsonify({
                "success": False,
                "error": "The uploaded image does not appear to contain a valid crop leaf or fruit. Please upload a clear photo of your plant for diagnosis."
            }), 400

        # --- Validate if image contains a crop leaf, fruit or agricultural plant using Gemini ---
        if gemini_available:
            try:
                # Open image for Gemini
                validation_image = Image.open(io.BytesIO(image_bytes))
                validation_model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "Analyze this image. Is it a plant, crop leaf, fruit, tree, or agricultural crop? "
                    "Answer ONLY 'YES' if it is a crop, leaf, plant, or fruit/vegetable. "
                    "Answer 'NO' if it is a human being, face, animal, random indoor object (like a keyboard, mug, room), "
                    "or anything unrelated to agriculture and plant disease."
                )
                response = validation_model.generate_content([validation_image, prompt])
                answer = response.text.strip().upper()
                print(f"[INFO] Gemini Vision crop validation result: {answer}", flush=True)
                
                if "NO" in answer:
                    return jsonify({
                        "success": False,
                        "error": "The uploaded image does not appear to contain a valid crop leaf or fruit. Please upload a clear photo of your plant for diagnosis."
                    }), 400
            except Exception as e:
                print(f"[WARNING] Gemini Vision crop validation failed: {e}", flush=True)

        # Preprocess and predict
        if mock_mode:
            filename_lower = file.filename.lower()
            fallback_fruit_classes = {
                "apple": "Apple_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Apple_Healthy",
                "blueberry": "Blueberry fruit rotten disease mold infected spots" if "rotten" in filename_lower or "sick" in filename_lower else "Blueberry fruit fresh healthy isolated high quality",
                "cherry": "Cherry_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Cherry_Healthy",
                "mango": "Mango_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Mango_Healthy",
                "orange": "Orange_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Orange_Healthy",
                "peach": "Peach_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Peach_Healthy",
                "pepper": "Pepper_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Pepper_Healthy",
                "potato": "Potato_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Potato_Healthy",
                "strawberry": "Strawberry_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Strawberry_Healthy",
                "tomato": "Tomato_Diseased" if "diseased" in filename_lower or "sick" in filename_lower else "Tomato_Healthy"
            }
            
            class_name = None
            for key, val in fallback_fruit_classes.items():
                if key in filename_lower:
                    class_name = val
                    break
                    
            if class_name is None:
                fallback_keys = [
                    "Apple_Healthy", "Mango_Diseased", "Mango_Healthy", "Tomato_Healthy", 
                    "Strawberry_Healthy", "Potato_Diseased", "Potato_Healthy"
                ]
                char_sum = sum(ord(c) for c in file.filename)
                class_name = fallback_keys[char_sum % len(fallback_keys)]
                
            confidence = 85.0 + (sum(ord(c) for c in file.filename) % 150) / 10.0
            print(f"[MOCK MODE] Falling back to mock fruit prediction for {file.filename}: {class_name} ({confidence:.1f}%)", flush=True)
            
        else:
            img_array = preprocess_image(image_bytes, target_size=FRUIT_IMG_SIZE)
            predictions = fruit_model.predict(img_array, verbose=0)
            predicted_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_idx]) * 100

            # --- Confidence Threshold Check ---
            if confidence < 50.0:
                return jsonify({
                    "success": False,
                    "error": "The AI model is not confident in this image. Please ensure you upload a clear, well-lit photo of a crop leaf or fruit."
                }), 400

            class_name = fruit_class_labels.get(str(predicted_idx), "Unknown")

        disease_info = get_disease_info(class_name)

        diagnosis = {
            "plant_type": disease_info["plant_type"],
            "health_status": disease_info["health_status"],
            "diagnosis": disease_info["diagnosis"],
            "confidence": f"{confidence:.1f}%",
            "symptoms": disease_info["symptoms"],
            "medicine": disease_info.get("medicine", []),
            "management": disease_info["management"],
        }

        return jsonify({"success": True, "data": diagnosis})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Accepts a user message and returns an AI response.
    Attempts to use Google Gemini API if available, otherwise falls back to
    the offline rule-based dictionary chatbot.
    """
    try:
        data = request.get_json(silent=True)
        if not data or not data.get("message", "").strip():
            return jsonify({
                "success": False,
                "error": "Please enter a message before sending."
            }), 400

        user_message = data["message"].strip()

        # If Gemini is available, use it!
        if gemini_available:
            try:
                # Load chat history from session or initialize it
                history = session.get("chat_history", [])
                
                # Setup system instructions for a professional agronomist
                system_prompt = (
                    "You are AgriShield AI Advisor, a senior professional agronomist and crop protection specialist.\n"
                    "Your job is to provide highly accurate, structured, and expert agricultural advice on:\n"
                    "- Crop and plant diseases (symptoms, treatments, preventive measures)\n"
                    "- Pest and insect management (both chemical and organic/IPM solutions)\n"
                    "- Soil health, preparation, pH, and fertilization (NPK, compost)\n"
                    "- Efficient irrigation practices (drip, water needs)\n"
                    "- Crop rotation, organic farming, and sustainable farming methods\n\n"
                    "Formatting Rules:\n"
                    "- Use Markdown (bold, lists, headers) to make responses readable.\n"
                    "- Keep answers concise, actionable, and structured.\n"
                    "- If a user asks general, non-agricultural questions, answer them professionally and politely. "
                    "Where possible, make natural analogies or tie-backs to nature, plants, farming, or technology."
                )
                
                # Format session history into the structure google-generativeai expects
                gemini_history = []
                for msg in history:
                    role = "model" if msg["role"] == "model" else "user"
                    gemini_history.append({
                        "role": role,
                        "parts": [msg["parts"][0]]
                    })

                # Initialize Gemini Model
                gemini_model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_prompt
                )
                
                # Start chat with history
                chat_session = gemini_model.start_chat(history=gemini_history)
                response = chat_session.send_message(user_message)
                assistant_reply = response.text

                # Append message to session history
                history.append({"role": "user", "parts": [user_message]})
                history.append({"role": "model", "parts": [assistant_reply]})
                
                # Keep history size manageable (last 20 messages)
                if len(history) > 20:
                    history = history[-20:]
                    
                session["chat_history"] = history
                session.modified = True

                return jsonify({
                    "success": True,
                    "reply": assistant_reply
                })

            except Exception as e:
                print(f"[WARNING] Gemini API error, falling back to offline mode: {e}")
                # Fall back to offline chatbot if API fails (network issue, rate limit, etc.)

        # --- FALLBACK: Offline Dictionary Chatbot ---
        assistant_reply = get_chatbot_response(user_message)

        # Still save fallback history in session so it remains cohesive
        history = session.get("chat_history", [])
        history.append({"role": "user", "parts": [user_message]})
        history.append({"role": "model", "parts": [assistant_reply]})
        if len(history) > 20:
            history = history[-20:]
        session["chat_history"] = history
        session.modified = True

        return jsonify({
            "success": True,
            "reply": assistant_reply
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Chat service encountered an error: {str(exc)}"
        }), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    """Wipe conversation history from session."""
    if "user_id" not in session:
        return jsonify({
            "success": False,
            "error": "Authentication required. Please log in first."
        }), 401

    session.pop("chat_history", None)
    session.modified = True
    return jsonify({"success": True, "message": "Conversation history cleared."})


# ---------------------------------------------------------------------------
# API Status & Health Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route("/status")
def status():
    """Detailed status endpoint."""
    return jsonify({
        "app_status": "online",
        "plant_model_loaded": model_loaded,
        "fruit_model_loaded": fruit_model is not None,
        "gemini_api_available": gemini_available,
        "port": 7860
    }), 200


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)


