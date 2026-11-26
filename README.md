# Ganado Protegido · Pronóstico Agroclimático a 3 km (GAIA Lab)

Sistema de alerta temprana y soporte a la decisión para criadores ganaderos altoandinos (Áncash, Perú). Combina pronósticos meteorológicos globales GFS (~27 km) con un modelo de *downscaling* neuronal a **3 km** para anticipar heladas nocturnas, orientar el pastoreo diurno y evaluar el riesgo de propagación de quemas.

## Integrantes

* Yeremi Alexander Camacho Solis
* Gabriel Dimas Gordillo Rodríguez 
* Jordy Joseph Valles Moya
* Alvaro Rodrigo Ayon Chinga

## Video de presentación

* **Enlace:** Pendiente

---

## Requisitos previos

* **Python 3.11 o superior** (versiones anteriores como 3.10 no son compatibles con las ruedas de `numpy==2.4.6`, `pandas==3.1.1`, `torch==2.15.0`, `matplotlib==3.11.2` y `streamlit==1.67.0`).
* **Git**
* Conexión a internet para descargar el pronóstico GFS en vivo desde Open-Meteo (la aplicación guarda caché local tras la primera ejecución).

---

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone [https://github.com/alexandersoliscama-coder/GAIA-Lab.git](https://github.com/alexandersoliscama-coder/GAIA-Lab.git)
cd GAIA-Lab
```

### 2. Crear y activar el entorno virtual (`venv`)

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Si PowerShell bloquea la ejecución de scripts, ejecuta primero `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` o usa CMD con `.venv\Scripts\activate.bat`).*

**En macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

Con el entorno virtual activado:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar la data de Kaggle en `data/`

La data no está en el repositorio por su tamaño. Puedes bajarla de dos formas (en ambos casos debes aceptar primero las reglas de la competencia en Kaggle):

* **Desde el navegador:** entra a la pestaña **Data** de la competencia en Kaggle, acepta las reglas, descarga los archivos y extrae el contenido dentro de la carpeta **`data/`**.
* **Desde la terminal (API de Kaggle):** desde la raíz del repositorio (`GAIA-Lab`), ejecuta:
  ```bash
  pip install kaggle
  kaggle competitions download -c hackathon-ia-pucp-2026
  unzip hackathon-ia-pucp-2026.zip -d data/
  ```
  *(En Windows PowerShell, si no tienes `unzip`, puedes descomprimir con `Expand-Archive hackathon-ia-pucp-2026.zip -DestinationPath data/`).*

Verifica que el archivo estático del dominio (`static.npz`) quede ubicado en **`data/`**:
```text
GAIA-Lab/
├── data/
│   └── static.npz
└── models/
    ├── modelo.pt
    ├── stats.npz
    └── chill_percentiles.npz
```
*(Los pesos y estadísticas del modelo en `models/` ya vienen incluidos en el repositorio).*

---

## Ejecutar la aplicación

Desde la raíz del repositorio (`GAIA-Lab`) y con el entorno virtual activo:

```bash
streamlit run app/streamlit_app.py
```

La interfaz se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## Estructura del proyecto

* `app/streamlit_app.py`: Interfaz web interactiva en Streamlit (tarjetas de decisión, comparación GFS vs. IA y mapas espaciales).
* `app/motor.py`: Lógica agroclimática y cálculo de umbrales (Senamhi-FAO, Nixon-Smith / FAO-56, Fosberg, ETCCDI y Pulgar Vidal).
* `app/gfs_forecast.py`: Descarga y construcción de ventanas temporales del pronóstico GFS desde Open-Meteo.
* `models/`: Pesos del modelo neuronal (`modelo.pt`), normalización (`stats.npz`) y percentiles históricos locales (`chill_percentiles.npz`).
* `data/`: Grilla estática topográfica (`static.npz`, descargada desde Kaggle).