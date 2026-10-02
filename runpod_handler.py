import os
import sys

# Desativa o sistema Xet do HuggingFace (causador do 'File reconstruction error')
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import time
import shutil
import base64
import requests
import subprocess
import traceback
from pathlib import Path
import runpod
from huggingface_hub import snapshot_download

# Diretórios base
APP_DIR = Path("/app/LongCat-Video")
RUNPOD_VOLUME = Path("/runpod-volume")

# Determina o diretório persistente de pesos (prioriza Network Volume se montado)
if RUNPOD_VOLUME.exists():
    WEIGHTS_BASE = RUNPOD_VOLUME / "weights"
else:
    WEIGHTS_BASE = APP_DIR / "weights"

AVATAR_CHECKPOINT_DIR = WEIGHTS_BASE / "LongCat-Video-Avatar-1.5"
FOUNDATION_CHECKPOINT_DIR = WEIGHTS_BASE / "LongCat-Video"

def ensure_models_downloaded():
    """Garante que apenas os pesos estritamente necessários para o Avatar 1.5 sejam baixados."""
    WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)
    
    # 1. Base Foundation Model (Baixa APENAS Tokenizer, UMT5 e VAE - Ignora DiT base não usado)
    if not (FOUNDATION_CHECKPOINT_DIR / "tokenizer").exists():
        print(f"[RunPod Worker] Baixando componentes base essenciais (Tokenizer, Text Encoder, VAE) para {FOUNDATION_CHECKPOINT_DIR}...")
        snapshot_download(
            repo_id="meituan-longcat/LongCat-Video",
            allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*"],
            local_dir=str(FOUNDATION_CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("[RunPod Worker] Componentes base essenciais baixados com sucesso!")

    # 2. Avatar 1.5 Model (Baixa APENAS scheduler, INT8 DiT, LoRA distill, Whisper e Vocal Separator)
    if not (AVATAR_CHECKPOINT_DIR / "scheduler").exists():
        print(f"[RunPod Worker] Baixando pesos do Avatar 1.5 otimizados para {AVATAR_CHECKPOINT_DIR}...")
        snapshot_download(
            repo_id="meituan-longcat/LongCat-Video-Avatar-1.5",
            allow_patterns=[
                "scheduler/*",
                "base_model_int8/*",
                "lora/*",
                "whisper-large-v3/*",
                "vocal_separator/*"
            ],
            local_dir=str(AVATAR_CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("[RunPod Worker] Pesos do Avatar 1.5 prontos!")


def download_asset(url: str, target_path: Path):
    """Baixa arquivos remotos (imagem de referência ou áudio) via stream."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)
    return target_path


def process_video(job):
    """Handler oficial invocado por cada requisição do RunPod Serverless."""
    job_id = job.get("id", f"job_{int(time.time())}")
    job_input = job.get("input", {})
    
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")
    prompt = job_input.get("prompt", "A person speaking naturally with expressive lipsync and lifelike movements")
    resolution = job_input.get("resolution", "480p")
    return_base64 = job_input.get("return_base64", True)

    if not image_url or not audio_url:
        return {"error": "Parametros obrigatorios: 'image_url' e 'audio_url'"}

    # Cria diretório de trabalho isolado para este job
    job_dir = Path(f"/tmp/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = job_dir / "input_image.jpg"
    audio_path = job_dir / "input_audio.wav"
    input_json_path = job_dir / "avatar_input.json"
    output_dir = job_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Garante pesos carregados/baixados no volume
        ensure_models_downloaded()

        # Baixa imagem e áudio do job
        print(f"[{job_id}] Baixando assets de entrada...")
        download_asset(image_url, img_path)
        download_asset(audio_url, audio_path)

        # Monta o JSON exato exigido pelo LongCat Avatar 1.5
        payload = {
            "prompt": prompt,
            "cond_image": str(img_path),
            "cond_audio": {
                "person1": str(audio_path)
            }
        }
        with open(input_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Executa inferência do LongCat Avatar 1.5 com Step Distillation e INT8 Quantization
        cmd = [
            "python", "run_demo_avatar_single_audio_to_video.py",
            f"--checkpoint_dir={str(AVATAR_CHECKPOINT_DIR)}",
            "--stage_1=ai2v",
            f"--input_json={str(input_json_path)}",
            f"--output_dir={str(output_dir)}",
            f"--resolution={resolution}",
            "--use_distill",
            "--model_type=avatar-v1.5",
            "--use_int8"
        ]

        print(f"[{job_id}] Executando comando de inferencia: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"[{job_id}] Erro na inferencia:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            return {
                "status": "error",
                "message": "Falha na inferencia do LongCat",
                "stderr": result.stderr[-2000:],
                "stdout": result.stdout[-2000:]
            }

        # Procura o vídeo gerado na pasta de saída
        generated_videos = list(output_dir.glob("*.mp4"))
        if not generated_videos:
            return {
                "status": "error",
                "message": "Nenhum arquivo MP4 encontrado na pasta de saida",
                "stdout": result.stdout[-2000:]
            }

        output_video_path = generated_videos[0]
        video_size = output_video_path.stat().st_size
        print(f"[{job_id}] Video gerado com sucesso! Tamanho: {video_size} bytes ({output_video_path.name})")

        response = {
            "status": "success",
            "video_filename": output_video_path.name,
            "video_size_bytes": video_size
        }

        # Retorna o vídeo em base64 se solicitado (ideal para WhatsApp / n8n)
        if return_base64:
            with open(output_video_path, "rb") as vf:
                response["video_base64"] = base64.b64encode(vf.read()).decode("utf-8")

        return response

    except Exception as e:
        print(f"[{job_id}] Excecao nao tratada: {traceback.format_exc()}")
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

    finally:
        # Limpeza temporária do job
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    print("[RunPod Worker] Iniciando LongCat Avatar 1.5 Serverless Handler...")
    runpod.serverless.start({"handler": process_video})
