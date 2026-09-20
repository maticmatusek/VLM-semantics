import os
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

# ===================== KONFIGURACIJA =====================

# Vnesi svoj Gemini API key direktno v skripto.
# Primer oblike ključa: "AIza..."


if not API_KEY or API_KEY == "TUKAJ_VNESI_SVOJ_GEMINI_API_KEY":
    raise RuntimeError("V skripti nastavi API_KEY na svoj pravi Gemini API ključ.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-pro"

# Golden datoteka in odgovori modelov.
# Skripta jih poskusi najti v trenutni mapi, data/, out_jsons/ in podmapah out_jsons/.
GOLD_FILE = "golden_final_test.json"
MODEL_FILES = {
    "model_1": "EuroVLM-9B-Preview_golden_final_test.json",
    "model_2": "gemma-3-12b-it_golden_final_test.json",
    "model_3": "gemma-3-4b-it_golden_final_test.json",
    "model_4": "SVILA-1-4B_golden_final_test.json",
    "model_5": "SVILA-1-12B_golden_final_test.json",
}

MODEL_LABELS = {
    "model_1": "EuroVLM-9B-Preview",
    "model_2": "gemma-3-12b-it",
    "model_3": "gemma-3-4b-it",
    "model_4": "SVILA-1-4B",
    "model_5": "SVILA-1-12B",
}

# Slike so pri tebi navadno v data/SLO_images/...
IMAGE_ROOT = Path("data")

# Outputi
OUTPUT_CSV = "gemini_judge_results_golden_final_test.csv"
OUTPUT_JSON_SUFFIX = "_with_scores.json"

# Koliko primerov hočeš testirati; None = vsi.
MAX_EXAMPLES = None

# Če želiš ponovno ocenjevati že ocenjene odgovore, nastavi na True.
OVERWRITE_EXISTING_JUDGES = False

# Majhen premor med Gemini klici.
SLEEP_SECONDS = 0.5

# Prompt template
PROMPT_TEMPLATE = """
Deluješ kot strog, a pošten evalvator odgovorov na vprašanja o SLIKAH (LLM-as-a-judge).

DOBIŠ:
- opis naloge in vprašanje o sliki,
- sliko, na katero se navezujejo vprašanja in odgovori,
- referenčni "golden standard" odgovor,
- 5 kandidatnih odgovorov modelov na isto vprašanje.

TVOJA NALOGA:
1. Vsak kandidatni odgovor (model_1 ... model_5) oceni po 4 dimenzijah (0–2 točk vsaka):
   1. IZPOLNITEV NALOGE (fulfilment)
      - 0: zgreši bistvo ali odgovori na drugo temo;
      - 1: delno; manjkajo ključni deli ali je preveč ohlapen;
      - 2: v celoti odgovori na zahtevo.
   2. VIZUALNA UTEMELJENOST & SPECIFIČNOST (visual_grounding)
      - 0: generičen opis, sklicevanje na nevidne stvari;
      - 1: nekaj konkretnih sklicev na vidne elemente;
      - 2: jasne, specifične podrobnosti (položaj, velikost, barve, ostrina, prekrivanje).
   3. RAZLAGA & KOHERENCA (explanation)
      - 0: nelogično, protislovno, skokovito;
      - 1: razumljivo, a tanko ali delno nedosledno;
      - 2: jasna, povezovalna razlaga, ki podpira trditve.
   4. NADZOR HALUCINACIJ (hallucination_control)
      - 0: izmišljuje objekte/odnose/barve, ki niso na sliki;
      - 1: manjše pretiravanje ali ugibanje brez jasne označitve;
      - 2: brez halucinacij; negotovost je ustrezno označena.

2. HALUCINACIJE – POSEBNO PRAVILO:
   - Če je pri kateremkoli odgovoru prisotna "velika halucinacija"
     (npr. glavni subjekt slike je napačen, izmišljene dominantne barve ali simboli),
     to jasno zabeleži v polju "major_hallucination": true.
   - V tem primeru pri tem odgovoru NE DVIGUJ ocen previsoko.

3. IZBIRA NAJBOLJŠEGA ODGOVORA:
   - Upoštevaj vseh 5 odgovorov modelov.
   - Uporabi 4 dimenzije in sliko.
   - Izberi en odgovor kot objektivno najboljši kompromis med:
     izpolnitvijo naloge, vizualno utemeljenostjo, koherenco in minimalnimi halucinacijami.
   - Golden standard odgovor služi samo kot REFERENCA in NI med kandidati za "best_answer".
   - Vrni ID kandidata (npr. "model_1", "model_2", ...).

4. IZHODNA OBLIKA – POMEMBNO:
   - VRNI IZKLJUČNO VELJAVEN JSON (brez razlage, brez komentarjev, brez dodatnega teksta).
   - Struktura je natanko:

{{
  "best_answer_id": "<ID enega od kandidatov: model_1 ... model_5>",
  "scores": [
    {{
      "answer_id": "model_1",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }},
    {{
      "answer_id": "model_2",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }},
    {{
      "answer_id": "model_3",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }},
    {{
      "answer_id": "model_4",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }},
    {{
      "answer_id": "model_5",
      "fulfilment": ...,
      "visual_grounding": ...,
      "explanation": ...,
      "hallucination_control": ...,
      "major_hallucination": ...
    }}
  ]
}}

--------------------------------------------------
KONKRETNA NALOGA:

ID slike: {example_id}
Vprašanje ({question_key}): {question_text}

Golden standard odgovor:
{gold_answer}

Kandidatni odgovori:
- model_1 : {answer_model_1}
- model_2 : {answer_model_2}
- model_3 : {answer_model_3}
- model_4 : {answer_model_4}
- model_5 : {answer_model_5}
"""


# ===================== POMOŽNE FUNKCIJE =====================

def resolve_file(filename: str) -> Path:
    """Poišče datoteko po tipičnih lokacijah in po out_jsons podmapah."""
    p = Path(filename)
    candidates = [
        p,
        Path("data") / filename,
        Path("out_jsons_final") / filename,
        Path("out_jsons_final") / "google" / filename,
        Path("out_jsons_final") / "GaMS-Beta" / filename,
        Path("out_jsons_final") / "utter-project" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c

    matches = list(Path(".").glob(f"**/{filename}"))
    if matches:
        # izberi najkrajšo pot, da se izogneš backupom v globokih mapah
        return sorted(matches, key=lambda x: len(str(x)))[0]

    raise FileNotFoundError(
        f"Ne najdem datoteke: {filename}. Poskrbi, da je v trenutni mapi, data/, out_jsons/ ali eni od podmap out_jsons/."
    )


def resolve_image_path(record: dict) -> Path | None:
    """Podpira image_path ali image, string ali list."""
    image_field = record.get("image_path", record.get("image"))
    if isinstance(image_field, list):
        image_field = image_field[0] if image_field else None
    if not image_field:
        return None

    rel = str(image_field).replace("\\", "/")
    candidates = [
        Path(rel),
        IMAGE_ROOT / rel,
        Path("data") / rel,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return IMAGE_ROOT / rel


def get_record_id(record: dict) -> str:
    """Tvoji golden_final_test JSON-i uporabljajo image_id; starejši lahko uporabljajo id."""
    if "image_id" in record:
        return str(record["image_id"])
    if "id" in record:
        return str(record["id"])
    raise KeyError("Zapis nima niti 'image_id' niti 'id'.")


def load_json(path_or_name):
    path = resolve_file(path_or_name)
    print(f"[INFO] Nalagam JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Prebranih zapisov iz {path}: {len(data)}")
    return data, path


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Shranil JSON: {path} (zapisov: {len(data)})")


def index_by_id(records):
    """records je list slovarjev; vrne dict image_id/id -> record."""
    result = {}
    for rec in records:
        rid = get_record_id(rec)
        if rid in result:
            raise ValueError(f"Duplikat id/image_id: {rid}")
        result[rid] = rec
    print(f"[DEBUG] index_by_id: ustvarjen index za {len(result)} id-jev")
    return result


def get_question_keys(record):
    """
    Vrne SLO in vse ključe tipa F1, F2, I1, A3 ...
    Ne vrača Answer*, *_suitable ali drugih metapodatkov.
    """
    pattern = re.compile(r"^[FIA]\d+$")
    keys = []
    if isinstance(record.get("SLO"), str):
        keys.append("SLO")
    keys.extend(k for k in record.keys() if pattern.match(k))

    def sort_key(k):
        if k == "SLO":
            return (0, "", 0)
        return (1, k[0], int(k[1:]))

    return sorted(keys, key=sort_key)


def build_prompt(example_id, question_key, question_text, gold_answer, model_answers):
    return PROMPT_TEMPLATE.format(
        example_id=example_id,
        question_key=question_key,
        question_text=question_text,
        gold_answer=gold_answer,
        label_model_1=MODEL_LABELS["model_1"],
        label_model_2=MODEL_LABELS["model_2"],
        label_model_3=MODEL_LABELS["model_3"],
        label_model_4=MODEL_LABELS["model_4"],
        label_model_5=MODEL_LABELS["model_5"],
        answer_model_1=model_answers["model_1"],
        answer_model_2=model_answers["model_2"],
        answer_model_3=model_answers["model_3"],
        answer_model_4=model_answers["model_4"],
        answer_model_5=model_answers["model_5"],
    )


def call_gemini_with_image(prompt, image_path=None):
    """Pošlje prompt in sliko Gemini-ju z novim google-genai SDK ter razčleni JSON odgovor."""
    contents = [prompt]

    if image_path is not None and Path(image_path).is_file():
        print(f"[INFO] Nalagam sliko za Gemini: {image_path}")
        image = Image.open(image_path).convert("RGB")
        contents.append(image)
    else:
        print(f"[WARN] Slika ne obstaja ali ni podana: {image_path}")

    print("[DEBUG] Pošiljam zahtevek Gemini-ju...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    text = (response.text or "").strip()
    print("[DEBUG] Surov odziv (prvih 500 znakov):")
    print(text[:500])

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        print("[DEBUG] JSON uspešno razčlenjen.")
        return parsed
    except json.JSONDecodeError:
        print("[WARN] JSONDecodeError, poskusim izrezati notranji JSON {...}")
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            parsed = json.loads(candidate)
            print("[DEBUG] JSON uspešno razčlenjen po izrezu.")
            return parsed
        print("[ERROR] Ni uspelo izluščiti veljavnega JSON-a iz odziva.")
        raise


def normalize_score(value, default=0):
    try:
        value = int(value)
    except Exception:
        return default
    return max(0, min(2, value))


# ===================== GLAVNI DEL =====================

def main():
    print("=== ZAČETEK EVALVACIJE GEMINI LLM-AS-A-JUDGE ===")

    # 1) Preberi vse JSONe
    gold_data, gold_path = load_json(GOLD_FILE)
    model_data = {}
    model_paths = {}
    for name, filename in MODEL_FILES.items():
        data, path = load_json(filename)
        model_data[name] = data
        model_paths[name] = path

    # 2) Naredi index po image_id/id
    gold_by_id = index_by_id(gold_data)
    model_by_id = {name: index_by_id(data) for name, data in model_data.items()}

    print(f"[INFO] Skupno število ID-jev v golden setu: {len(gold_by_id)}")
    if MAX_EXAMPLES is not None:
        print(f"[INFO] TEST MODE: obdelam samo prvih {MAX_EXAMPLES} ID-jev.")
    else:
        print("[INFO] FULL MODE: obdelam vse ID-je.")

    all_rows = []

    items = list(gold_by_id.items())
    if MAX_EXAMPLES is not None:
        items = items[:MAX_EXAMPLES]

    for idx, (rid, gold_rec) in enumerate(items, start=1):
        print(f"\n=== Primer {idx}/{len(items)} – image_id/id={rid} ===")

        question_keys = get_question_keys(gold_rec)
        print(f"[DEBUG] Najdena vprašanja za id={rid}: {question_keys}")

        image_path = resolve_image_path(gold_rec)
        print(f"[DEBUG] image_path={image_path}")

        for qk in question_keys:
            judge_key = qk + "_judge"

            # Če vsi modeli že imajo oceno in nočeš overwrite, preskoči.
            if not OVERWRITE_EXISTING_JUDGES:
                already_done = True
                for mname in MODEL_FILES.keys():
                    rec_m = model_by_id[mname].get(rid)
                    if rec_m is None or judge_key not in rec_m:
                        already_done = False
                        break
                if already_done:
                    print(f"  -> Preskakujem {qk}, ker ocene že obstajajo.")
                    continue

            print(f"\n  -> Obdelujem vprašanje: {qk}")
            answer_key = "Answer" + qk
            question_text = gold_rec.get(qk, "")
            gold_answer = gold_rec.get(answer_key, "")

            if not isinstance(question_text, str) or not question_text.strip():
                print(f"[WARN] Prazno vprašanje {qk} pri id={rid}, preskočim.")
                continue

            print(f"     Vprašanje: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")
            print(f"     Golden odgovor: {str(gold_answer)[:120]}{'...' if len(str(gold_answer)) > 120 else ''}")

            # Zberi kandidatne odgovore vseh modelov
            model_answers = {}
            missing = False
            for mname in MODEL_FILES.keys():
                rec_m = model_by_id[mname].get(rid)
                if rec_m is None:
                    print(f"[WARN] id {rid} manjka v {mname}, skačem to vprašanje.")
                    missing = True
                    break

                model_answer = rec_m.get(answer_key, "")
                if not isinstance(model_answer, str):
                    model_answer = str(model_answer)
                model_answers[mname] = model_answer
                print(f"     {mname} ({MODEL_LABELS[mname]}) odgovor: {model_answer[:120]}{'...' if len(model_answer) > 120 else ''}")

            if missing:
                continue

            prompt = build_prompt(
                example_id=rid,
                question_key=qk,
                question_text=question_text,
                gold_answer=gold_answer,
                model_answers=model_answers,
            )
            print(f"[DEBUG] Prompt dolg: {len(prompt)} znakov")

            try:
                result = call_gemini_with_image(prompt, image_path=image_path)
            except Exception as e:
                print(f"[ERROR] Gemini fail pri id={rid}, q={qk}: {e}")
                continue

            best_id = result.get("best_answer_id")
            scores_list = result.get("scores", [])

            print(f"[INFO] best_answer_id za id={rid}, q={qk}: {best_id}")
            print(f"[DEBUG] Prejeto število ocen: {len(scores_list)}")

            for score_entry in scores_list:
                answer_id = score_entry.get("answer_id")
                if answer_id not in MODEL_FILES:
                    print(f"[WARN] Neznan answer_id v Gemini odgovoru: {answer_id}, preskočim.")
                    continue

                fulfilment = normalize_score(score_entry.get("fulfilment", 0))
                visual_grounding = normalize_score(score_entry.get("visual_grounding", 0))
                explanation = normalize_score(score_entry.get("explanation", 0))
                hallucination_control = normalize_score(score_entry.get("hallucination_control", 0))
                major_hall = bool(score_entry.get("major_hallucination", False))

                total = fulfilment + visual_grounding + explanation + hallucination_control
                is_best = answer_id == best_id

                print(
                    f"     Ocena za {answer_id} ({MODEL_LABELS[answer_id]}): "
                    f"F={fulfilment}, VG={visual_grounding}, E={explanation}, H={hallucination_control}, "
                    f"MAJOR={major_hall}, TOTAL={total}, BEST={is_best}"
                )

                row = {
                    "id": rid,
                    "image_id": rid,
                    "subquestion": qk,
                    "model": answer_id,
                    "model_label": MODEL_LABELS[answer_id],
                    "fulfilment": fulfilment,
                    "visual_grounding": visual_grounding,
                    "explanation": explanation,
                    "hallucination_control": hallucination_control,
                    "major_hallucination": major_hall,
                    "total": total,
                    "best": is_best,
                }
                all_rows.append(row)

                model_rec = model_by_id[answer_id][rid]
                model_rec[judge_key] = {
                    "model_label": MODEL_LABELS[answer_id],
                    "fulfilment": fulfilment,
                    "visual_grounding": visual_grounding,
                    "explanation": explanation,
                    "hallucination_control": hallucination_control,
                    "major_hallucination": major_hall,
                    "total": total,
                    "best_for_this_question": is_best,
                }

            time.sleep(SLEEP_SECONDS)

    # 6) Shrani CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n[OK] Rezultati zapisani v {OUTPUT_CSV} (vrstic: {len(df)})")

    # 7) Shrani posodobljene model JSONe v iste mape kot originali
    for mname, path in model_paths.items():
        orig_list = model_data[mname]
        out_path = Path(path).with_name(Path(path).stem + OUTPUT_JSON_SUFFIX)
        save_json(orig_list, out_path)
        print(f"[OK] Posodobljen JSON za {mname} ({MODEL_LABELS[mname]}) -> {out_path}")

    print("=== KONEC EVALVACIJE ===")


if __name__ == "__main__":
    main()
