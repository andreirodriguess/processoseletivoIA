# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Andrei Luiz da Silva Rodrigues**

### 1️⃣ Resumo da Abordagem

Foram utilizados os seguintes hiperparâmetros no fine-tuning:  

épocas: 30  
Tamanho da imagem: 640x640  
batch size: 16  
cls : 0.8  
cls_pw : 0.5  


Para lidar com o desbalanceamento de classes no dataset foram ajustados os hiperparâmetros:  
 Peso da perda de classificação('cls'), para aumentar a penalização do modelo em caso de erro de classe,  
 e o Power for class weighting('cls_pw') para aumentar a penalização em caso de erro nas classes minoritárias.


### 2️⃣ Bibliotecas Utilizadas

As bibliotecas utilizadas estão listadas a seguir: 

ultralytics  
torch e torchvision  
opencv-python(cv2)  
tensorflow  

### 3️⃣ Técnica de Otimização do Modelo  

A otimização estrutural do modelo foi feita ao executar o script 'optimize_model.py'. Ao converter o modelo PyTorch estruturado em ponto flutuante de 32 bits('.pt') para o formato TensorFlow Lite('.tflite') usando o ecossistema de compilação da Google ('LiteRT'), por meio da chamada nativa da biblioteca Ultralytics (`model.export(format="tflite")`).

### 4️⃣ Resultados Obtidos

Após 30 épocas de fine-tuning, o modelo obteve os seguintes resultados de acurácia no conjunto de validação, de acordo com o arquivo ('results.png'):

mAP50 : [aproximadamente 81%]  
mAP50-95 : [aproximadamente 57%]

O tamanho final do arquivos:

Tamanho do arquivo `model.pt`: 5.38 KB  
Tamanho do arquivo `model.tflite`: 10.38 KB


### 5️⃣ Comentários Adicionais (Opcional)

A principal dificuldade encontrada foi a necessidade de lidar com o dataset com classes desbalanceadas no treinamento do modelo. Para tal, foi necessário consultar a documentação oficial da biblioteca Ultralytics e obter os parametros que me ajudariam a lidar com tal problemática.

### 6️⃣ Exemplo de Inferência

```text
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 15 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss105.jpg                          9  [9x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss107.jpg                          1  [1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss11.jpg                          31  [1x mask_weared_incorrect, 29x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss113.jpg                          4  [3x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss12.jpg                          13  [12x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss123.jpg                          2  [2x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss124.jpg                          6  [3x with_mask, 3x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss126.jpg                          3  [2x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss128.jpg                          1  [1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss129.jpg                          3  [1x with_mask, 2x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss130.jpg                          5  [3x without_mask, 1x mask_weared_incorrect, 1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss135.jpg                          5  [3x with_mask, 2x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss139.jpg                         19  [1x mask_weared_incorrect, 17x with_mask, 1x without_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss142.jpg                          1  [1x with_mask]
Results saved to /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss143.jpg                          1  [1x with_mask]
----------------------------------------------------------------------
TOTAL                                      104

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
   (Abra essa pasta para verificar visualmente as bounding boxes preditas)
```

Nota: As caixas ficaram bem localizadas na maioria das imagens, porém, houveram raras exceções em que o modelo errou, e houve muita eficiência do modelo na detecção das classes corretamente, não foi possível perceber um desempenho inferior para a classe minoritária(`mask_weared_incorrect`).

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
