from pathlib import Path
import json

import pandas as pd

from src.skills_reference import SkillReference
from src.comparator import compare_texts

DATA_DIR = Path(__file__).parent / "data"

def main():
    skill_ref = SkillReference.load(DATA_DIR / "skills_reference.json")
    df = pd.read_excel(DATA_DIR / "huggingface_resume_job_fit_RAW.xlsx")
    row = df.sample(1).iloc[0]
    res = compare_texts(cv_text=str(row["resume_text"]), job_text=str(row["job_description_text"]), skill_ref=skill_ref)
    out = {
        "scores": res.scores,
        "missing_skills": res.missing_skills[:20],
        "matched_skills": res.matched_skills[:20],
        "notes": res.notes,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
