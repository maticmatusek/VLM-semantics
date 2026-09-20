
import os
import json
import random
import asyncio
import traceback

import nest_asyncio
nest_asyncio.apply()

from google import genai
from PIL import Image
from aiolimiter import AsyncLimiter

# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "gemini-2.5-flash"
INPUT_JSON = "unified_images_minimal.json"
OUTPUT_JSON = "golden_final.json"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 120
SAVE_EVERY = 1

# NEW: limit how many items/images/IDs to process
# None = process all
MAX_ITEMS = 5000

api_rate_limiter = AsyncLimiter(max_rate=10, time_period=1)

# =========================================================
# QUESTIONS
# =========================================================

SLO = [
    "Kaj ta slika pomeni za Slovenijo?",
    "Kakšen pomen ima ta slika za Slovenijo?",
    "Kaj predstavlja ta slika v slovenskem kontekstu?",
    "Kako je ta slika pomembna za Slovenijo?",
    "Kaj nam ta slika pove o Sloveniji?",
    "Kaj simbolizira ta slika za Slovenijo?",
    "Kako ta slika odraža pomen Slovenije?",
    "V čem je pomen te slike za Slovenijo?",
    "Zakaj je ta slika pomembna za Slovenijo?",
    "Kaj ta podoba predstavlja za slovensko identiteto?"
]

F_1 = [
    "Kateri del slike je v ospredju in zakaj je pomemben za zgodbo, ki jo slika pripoveduje?",
    "Opiši glavni motiv slike in elemente, ki ga podpirajo v ozadju.",
    "Kaj na sliki pritegne največ pozornosti in kako okolica to poudari?",
    "Katere podrobnosti so ključne za razumevanje pomena slike, in katere služijo le kot ozadje?",
    "Kako se razlikuje vloga glavnega lika od ostalih elementov na sliki?",
    "Kateri deli slike so bistveni za zgodbo, in kateri le ustvarjajo kontekst ali vzdušje?",
    "Kaj na sliki deluje kot glavni poudarek in kako kompozicija podpira to vlogo?",
    "Kako bi razdelil(a) sliko na pomembne (v ospredju) in podporne (v ozadju) elemente?",
    "Kaj je osrednja točka slike in katere vizualne značilnosti vodijo pogled gledalca tja?",
    "Kako ozadje prispeva k razumevanju pomena ali čustvenega tona glavnega motiva?"
]

F_2 = [
    "Opiši odnose med glavnimi predmeti na sliki – kako so med seboj povezani?",
    "Kako postavitev predmetov vpliva na razumevanje pomena ali dinamike prizora?",
    "Kateri predmet deluje kot glavni in kateri kot podporni – kako to vidiš iz kompozicije?",
    "Ali kateri izmed predmetov prevladuje nad drugimi ali je v podrejenem položaju?",
    "Kako razdalja, smer pogleda ali orientacija predmetov vpliva na razmerja med njimi?",
    "Kaj pomeni fizična bližina ali ločenost določenih elementov v prostoru slike?",
    "Ali predmeti na sliki sodelujejo, se nasprotujejo ali se dopolnjujejo – kako to vidiš?",
    "Kako njihova postavitev ustvarja občutek ravnotežja, napetosti ali kontrasta?",
    "Kateri odnosi med objekti so ključni za razumevanje zgodbe ali sporočila slike?",
    "Kako bi opisal(a) prostorsko in simbolno razmerje med glavnimi elementi v prizoru?"
]

F_3 = [
    "Kakšno razpoloženje izraža ta slika in kako k temu prispevajo uporabljene barve?",
    "Katere barve prevladujejo na sliki in kakšne občutke vzbujajo pri gledalcu?",
    "Kako barvna paleta vpliva na ton ali čustveno ozračje slike?",
    "Ali barve na sliki simbolizirajo določene ideje, čustva ali kulturne pomene?",
    "Kako bi se spremenilo razpoloženje slike, če bi bile uporabljene druge barve?",
    "Katere barve na sliki delujejo harmonično in katere ustvarjajo napetost?",
    "Ali so barve na sliki tople ali hladne – in kako to vpliva na čustveni vtis?",
    "Kako kontrasti med barvami oblikujejo razpoloženje prizora?",
    "Katera barva najbolj pritegne pozornost in kakšen pomen ima v kontekstu slike?",
    "Kaj sporoča kombinacija uporabljenih barv o vzdušju ali namenu slike?"
]

I_1 =[
    "Ali je kompozicija slike uravnotežena? Kaj je v njej poudarjeno?",
    "Kako uporaba prostora (praznina, razporeditev elementov) vpliva na občutek ravnotežja?",
    "Ali slika uporablja pravilo tretjin ali simetrijo – in kakšen učinek to ima?",
    "Kako postavitev glavnih elementov vpliva na vizualno dinamiko slike?",
    "Ali kompozicija deluje harmonično ali namenoma neuravnoteženo? Zakaj misliš tako?",
    "Kje se nahaja težišče slike in kako vpliva na to, kam gledalec najprej pogleda?",
    "Kako uporaba asimetrije prispeva k zanimivosti ali napetosti prizora?",
    "Ali prazni prostori v kompoziciji pomagajo poudariti glavni motiv ali ga oslabijo?",
    "Kako bi ocenil(a) estetsko skladnost razporeditve elementov na sliki?",
    "Kaj kompozicija sporoča o hierarhiji pomenov med posameznimi deli slike?"
]

I_2 = [
    "Kaj predstavlja glavni simbol ali predmet na sliki v kulturnem ali družbenem kontekstu?",
    "Ali prepoznaš kakšen znan simbol in kakšen pomen ima v tej upodobitvi?",
    "Kako bi razložil(a) pomen glavnega predmeta glede na kulturne ali zgodovinske okoliščine?",
    "Ali simbol na sliki nosi univerzalen pomen ali je vezan na določeno kulturo?",
    "Kako uporaba določenega simbola vpliva na sporočilo ali ton slike?",
    "Ali se pomen simbola spreminja glede na kontekst, v katerem je prikazan?",
    "Katere kulturne ali verske konotacije ima glavni predmet na sliki?",
    "Kako bi povprečen gledalec iz druge kulture razumel ta simbol?",
    "Ali slika uporablja simbol na pričakovan ali subverziven (preobraten) način?",
    "Kaj nakazuje izbira tega simbola o avtorjevem namenu ali družbenem komentarju?"
]

I_3 = [
    "Katero idejo ali sporočilo slika izraža na metaforičen način?",
    "Ali slika uporablja kakšen predmet ali prizor kot simbol za nekaj abstraktnega?",
    "Kaj bi lahko pomenila podoba, če jo razumeš kot prispodobo ali metaforo?",
    "Kateri elementi slike nakazujejo, da gre za simbolno ali metaforično upodobitev?",
    "Kako bi razložil(a) skriti pomen ali sporočilo, ki ga slika posreduje prek simbolike?",
    "Ali slika uporablja znane vizualne trope (npr. sence kot dvojniki, ptice kot svoboda)?",
    "Kako se konkretni motivi na sliki spreminjajo v ideje ali čustva na simbolni ravni?",
    "Kateri del slike bi lahko razumeli kot prispodobo za notranje stanje ali družbeni komentar?",
    "Ali slika združuje resnične in domišljijske elemente za ustvarjanje metaforičnega pomena?",
    "Kaj želi avtor povedati s to vizualno prispodobo – katero širšo idejo ali občutek izraža?"
]

A_1 = [
    "Kaj ta predmet dobesedno prikazuje in kaj bi lahko simboliziral?",
    "Kako bi opisal(a) ta element na sliki na ravni denotacije (kaj vidimo) in konotacije (kaj pomeni)?",
    "Kaj je neposredni pomen upodobljenega predmeta in kakšen skriti pomen lahko nosi?",
    "Kateri čustveni ali simbolni pomeni se povezujejo z upodobljenim objektom?",
    "Ali ima ta predmet v določeni kulturi ali kontekstu poseben simbolni pomen?",
    "Kako se razlikuje med tem, kar slika prikazuje, in tem, kar sporoča?",
    "Kaj bi povsem nevtralen opazovalec rekel, da vidi – in kaj bi nekdo, ki pozna simboliko, razumel drugače?",
    "Ali je pomen predmeta zgolj vizualen ali prenaša tudi idejo, vrednoto ali čustvo?",
    "Kako se dobesedni videz predmeta uporablja za ustvarjanje simbolnega učinka?",
    "Kaj je na sliki prikazano dobesedno in kaj namiguje na globlji pomen ali zgodbo?"
]

A_2 = [
    "Na katero drugo umetnino ali medijsko podobo bi ta slika lahko namigovala?",
    "Ali slika vsebuje kakšne prepoznavne vizualne reference na znana dela ali motive iz popularne kulture?",
    "Kateri deli slike te spomnijo na druge znane podobe, prizore ali zgodbe?",
    "Ali bi lahko ta slika bila reinterpretacija ali parodija katerega znanega umetniškega dela?",
    "Kaj na sliki nakazuje, da se sklicuje na določen film, serijo, sliko ali spletni mem?",
    "Kateri kulturni ali zgodovinski motivi so prepoznavni v tej podobi?",
    "Ali obstaja znana zgodba ali lik, na katerega ta slika očitno ali subtilno namiguje?",
    "Kako ta slika vzpostavlja dialog z drugimi deli umetnosti ali mediji, ki jih poznaš?",
    "Katera znana vizualna kompozicija ali prizor bi lahko bil vir navdiha za to sliko?",
    "Ali prepoznaš kakšne skrite simbole ali citate iz drugih umetniških del v tej sliki?"
]

A_3 = [
    "Katero sporočilo želi ta slika posredovati in komu je namenjeno?",
    "Kdo je verjetno ciljno občinstvo te slike in zakaj misliš tako?",
    "Kakšen vtis ali čustveni odziv želi slika povzročiti pri gledalcu?",
    "Ali ima slika prepričevalni namen – če da, v katero smer poskuša vplivati?",
    "Kateri elementi slike nakazujejo, komu je namenjena (npr. starost, interesi, vrednote)?",
    "Kaj želi avtor slike, da si gledalec misli, čuti ali naredi po ogledu slike?",
    "Kako slog, barve in kompozicija prispevajo k sporočilu slike?",
    "Ali bi to sporočilo delovalo enako na vse gledalce ali le na določeno skupino?",
    "Kakšen je namen slike?",
    "Kaj pove izbira vizualnih simbolov o ciljni skupini, ki jo slika nagovarja?"
]

# =========================================================
# HELPERS
# =========================================================

def clean_json_text(text):
    text = (text or "").replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text

def choose_questions():
    nums = [random.randint(0, 9) for _ in range(10)]
    return {
        "SLO": SLO[nums[9]],
        "F1": F_1[nums[0]],
        "F2": F_2[nums[1]],
        "F3": F_3[nums[2]],
        "I1": I_1[nums[3]],
        "I2": I_2[nums[4]],
        "I3": I_3[nums[5]],
        "A1": A_1[nums[6]],
        "A2": A_2[nums[7]],
        "A3": A_3[nums[8]],
    }

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_image_path(item):
    candidates = [
        item.get("saved_image"),
        item.get("image_path"),
        item.get("image"),
        item.get("local_path"),
    ]

    for c in candidates:
        if not c:
            continue

        if isinstance(c, list):
            if len(c) == 0:
                continue
            c = c[0]

        c = str(c).strip()

        if os.path.isdir(c):
            continue

        if os.path.isfile(c):
            return c

    return ""

def ensure_questions_exist(item):
    if "SLO" not in item or not item["SLO"]:
        q = choose_questions()
        item["SLO"] = q["SLO"]
        item["F1"] = q["F1"]
        item["F2"] = q["F2"]
        item["F3"] = q["F3"]
        item["I1"] = q["I1"]
        item["I2"] = q["I2"]
        item["I3"] = q["I3"]
        item["A1"] = q["A1"]
        item["A2"] = q["A2"]
        item["A3"] = q["A3"]

def batch_done(item, batch_keys):
    for key in batch_keys:
        answer_key = "Answer" + key
        suitable_key = None if key == "SLO" else key + "_suitable"

        if answer_key not in item or item[answer_key] in [None, ""]:
            return False

        if suitable_key is not None:
            if suitable_key not in item or item[suitable_key] in [None, ""]:
                return False

    return True

# =========================================================
# MAIN ASYNC FUNCTION
# =========================================================

async def process_batch(item, batch_keys, delay=1.5):
    client = genai.Client(api_key=GOOGLE_API_KEY)

    item_id = item.get("index", item.get("id", item.get("image_id", "")))
    image = get_image_path(item)
    text = item.get("image_caption", item.get("image_description", item.get("description", "")))
    text1 = item.get("article_text", item.get("text", item.get("context", "")))

    fields = []
    if "text_image_correlation" not in item or item.get("text_image_correlation") in [None, ""]:
        fields.append('- "text_image_correlation": good, medium, bad')

    for key in batch_keys:
        fields.append(f'- "{key}": {item[key]}')
        fields.append(f'- "Answer{key}": fill in the answer')
        if key != "SLO":
            fields.append(f'- "{key}_suitable": fill in the answer')

    fields_text = "\n".join(fields)

    prompt = f"""Odgovori na vprašanja o tej sliki, pri tem pa upoštevaj tudi dodatno besedilo: {text}

Vrni slovar z naslednjimi vrhnjimi ključi:
{fields_text}

Dosledno upoštevaj naslednjo strukturo in jezikovne zahteve:
- Odgovori morajo biti v slovenščini.
- Besedila ne omenjaj neposredno, ampak ga uporabi za oblikovanje pravilnih odgovorov.
- Pod zahtevanimi ključi zapiši samo vprašanje.
- Pod suitable presodi, kako primerno je vprašanje za sliko, in kot odgovor zapiši da, srednje ali ne.
- Pod text_image_correlation presodi, kako primerno je besedilo za sliko, in kot odgovor zapiši dobro, srednje ali slabo.
- Vrni IZKLJUČNO veljaven JSON.
- Narekovaje znotraj nizov pravilno ubeži.
"""

    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            print("=" * 80, flush=True)
            print(f"[TRY] item_id={item_id} | batch={batch_keys} | attempt {attempt + 1}/{MAX_RETRIES}", flush=True)
            print("[DEBUG] image path:", image, flush=True)
            print("[DEBUG] image exists:", os.path.exists(image), flush=True)

            await asyncio.sleep(delay)

            async with api_rate_limiter:
                print("[DEBUG] calling Gemini...", flush=True)

                with Image.open(image) as img:
                    result = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=MODEL_NAME,
                            contents=[prompt, img]
                        ),
                        timeout=REQUEST_TIMEOUT
                    )

            print("[DEBUG] got response", flush=True)

            text_out = result.text
            print("[DEBUG] result.text type:", type(text_out).__name__, flush=True)
            print("[DEBUG] first 300 chars:", str(text_out)[:300], flush=True)

            processed_text = json.loads(clean_json_text(text_out))
            return processed_text

        except Exception as e:
            print("EXCEPTION:", repr(e), flush=True)
            traceback.print_exc()
            attempt += 1

            if attempt >= MAX_RETRIES:
                return {"generation_error": f"Error after {MAX_RETRIES} retries on batch {batch_keys}: {e}"}

# =========================================================
# RUN
# =========================================================

if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your terminal first.")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

if os.path.exists(OUTPUT_JSON):
    print(f"[INFO] Found existing {OUTPUT_JSON}, loading it for resume...", flush=True)
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    if isinstance(saved_data, list) and len(saved_data) == len(data):
        data = saved_data
        print("[INFO] Resume mode enabled.", flush=True)
    else:
        print("[WARNING] Existing output length does not match input. Starting from input file.", flush=True)

BATCHES = [
    ["SLO", "F1"],
    ["F2", "F3"],
    ["I1", "I2"],
    ["I3", "A1"],
    ["A2", "A3"],
]

counter = 0
processed_items = 0

for item in data:
    counter += 1

    if MAX_ITEMS is not None and processed_items >= MAX_ITEMS:
        print(f"[STOP] Reached MAX_ITEMS={MAX_ITEMS}", flush=True)
        break

    ensure_questions_exist(item)

    item_id = item.get("index", item.get("id", item.get("image_id", counter)))
    image = get_image_path(item)

    print("\n" + "#" * 80, flush=True)
    print(f"[ITEM] {counter}/{len(data)} | id={item_id}", flush=True)
    print(f"[ITEM] image={image}", flush=True)

    if not image or not os.path.isfile(image):
        print("[ERROR] image missing or not a file, skipping", flush=True)
        item["generation_error"] = f"Image missing or not a file: {image}"
        if counter % SAVE_EVERY == 0:
            save_json(data, OUTPUT_JSON)
            print("[SAVE] progress saved", flush=True)
        continue

    item_was_processed = False

    for batch_keys in BATCHES:
        if batch_done(item, batch_keys):
            print(f"[SKIP] batch already done: {batch_keys}", flush=True)
            continue

        answers = asyncio.get_event_loop().run_until_complete(
            process_batch(item, batch_keys, delay=2)
        )

        if isinstance(answers, dict):
            item.update(answers)
        else:
            item["generation_error"] = f"Unexpected output for batch {batch_keys}: {answers}"

        item_was_processed = True

        print(f"[DONE] batch processed: {batch_keys}", flush=True)

        if counter % SAVE_EVERY == 0:
            save_json(data, OUTPUT_JSON)
            print("[SAVE] progress saved", flush=True)

    if item_was_processed:
        processed_items += 1
        print(f"[COUNT] processed_items={processed_items}", flush=True)

print("\n[FINAL SAVE] writing output...", flush=True)
save_json(data, OUTPUT_JSON)
print("[DONE] all finished", flush=True)