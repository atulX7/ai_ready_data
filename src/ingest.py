import os
import requests

# Ensure raw data folder exists
os.makedirs("data/raw", exist_ok=True)

def download(name, url, dest_path):
    try:
        print(f"🔽 Downloading {name}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Saved to {dest_path}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

def generate_noisy_data(file_path):
    print("⚠️  Generating simulated non-AI-ready text...")
    noisy_text = """
    ##DrUg 123! - Use4 heartattack,,sometimes BP.   dont take milk.!! @@@
    u may feel dizZzy or not. SideEffx: headache, /vomt/, or bloooood issues
    check with Doc!! maybe. REF#244; u know???
    ...
    (more garbled text here to confuse any AI system trying to read)
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(noisy_text.strip())
    print(f"✅ Non-AI-ready file created at {file_path}")

# ✅ Download AI-ready label (structured medical content)
ai_ready_url = "https://dailymed.nlm.nih.gov/dailymed/archives/fdaDrugLabel.cfm?archiveid=658328"
ai_ready_path = "data/raw/cymbalta.txt"
download("cymbalta", ai_ready_url, ai_ready_path)

# ✅ Create noisy text locally instead of downloading
non_ai_ready_path = "data/raw/noisy_drug_info.txt"
generate_noisy_data(non_ai_ready_path)

print("📁 Ingestion complete.")

