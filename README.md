# tfg_steel_plate_defects_v1
Sigue estos pasos para clonar el repositorio, instalar las dependencias y ejecutar el proyecto:

```
# 1. Clonar el repositorio
git clone https://github.com/ignagp34/tfg_steel_plate_defects_v1.git
cd tfg_steel_plate_defects_v1

# 2. Crear el entorno e instalar dependencias (usando conda)
conda env create -f environment.yml
conda activate TFG

# 3. Ejecutar el código principal
python main.py

# (Opcional) Abrir el notebook
jupyter notebook
```

---

## Descarga de modelos y datos

1. Seguir los notebooks o utilizar comandos p.e: "python -m src.cli.train --n-splits 5 --model-name lgbm --no-oversample"
2. \src\visualization\eda.py  para generar gráficas

---
## Requisitos previos

- Git
- Conda (Anaconda o Miniconda)
---
