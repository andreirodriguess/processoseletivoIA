import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

# insira seu código aqui


model = YOLO("yolo11n.pt")
results = model.train(
    data="dataset/data.yaml",
    epochs=1,
    imgsz=640,
    batch=16,
    device="cpu",
    
    #ajustes para lidar com o desbalanceamento de classes

    cls_pw=0.5, #aumento da penalização por erro de classe minoritária
    cls=0.8 #aumento da penalidade por erro de classificação
)
shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")
