# 📦 Ce se dă

    🖼️ Set de date cu imagini mamografice, fiecare etichetată cu tipul tumorii (benignă sau malignă). Setul poate conține și metadate relevante, cum ar fi vârsta pacientei, densitatea țesutului sau poziția tumorii.

    🤖 Modele AI pre-antrenate din familia Transformer (ex: Vision Transformer – ViT), pregătite pentru fine-tuning specific pe domeniul medical.

    🛠️ Mediu software cu Python și librării specializate: PyTorch, TensorFlow, instrumente pentru augmentare a datelor și evaluare metrică.

# 🎯 Ce se cere

Dezvoltarea unui sistem inteligent care să analizeze automat mamografii și să clasifice tumorile ca benigne sau maligne, pentru diagnosticare timpurie a cancerului de sân.

Obiective specifice:

    🧹 Preprocesarea și pregătirea imaginilor pentru rețele neuronale.

    ⚙️ Alegerea și fine-tuning-ul unui model Transformer pe setul de date.

    📊 Antrenarea și evaluarea modelului cu metrici medicale relevante: acuratețe, precizie, sensibilitate, specificitate și AUC-ROC.

    🚫 Reducerea erorilor de clasificare, în special a falselor negative, cu impact critic asupra pacientelor.

    🤝 Oferirea unui pipeline automatizat și interpretabil care să sprijine medicii în luarea deciziilor clinice, îmbunătățind viteza și calitatea diagnosticării.


🩺 Identificarea Cancerului de Sân pe Baza Mamografiilor

Detecția cancerului de sân reprezintă o sarcină extrem de complexă și sensibilă, unde acuratețea diagnosticului este esențială pentru salvarea vieților. Utilizarea inteligenței artificiale (AI), în special a modelelor moderne bazate pe Transformers, este justificată prin multiple motive fundamentale:

# ⚠️ 1. Limitările evaluării umane

🔄 Variabilitate între radiologi: Interpretările pot diferi semnificativ între medici și pot fi influențate de oboseală, nivel de experiență sau alți factori subiectivi.

❗ Rate mari de rezultate false:

Fals pozitive → pot duce la intervenții medicale inutile.

Fals negative → pot duce la întârzieri periculoase în tratament.

# 🧬 2. Complexitatea imaginilor mamografice

🧱 Variație mare în densitatea sânului: Țesutul glandular dens poate ascunde leziuni suspecte.

🧩 Structuri suprapuse: Tumorile pot fi mascate de alte structuri anatomice normale.

🔬 Diferențe subtile între leziuni benigne și maligne: Acestea sunt dificil de distins, chiar și pentru radiologi cu experiență.

# ⚡ 3. Scalabilitate și viteză

⏱️ Analiză rapidă a mii de imagini: AI poate procesa volume mari de date într-un timp foarte scurt.

🧠 Asistență în triere: Prioritizarea automată a cazurilor cu probabilitate mare de malignitate.

🚀 Reducerea timpului de diagnostic: Rezultatele pot fi livrate rapid pentru a fi revizuite de specialiști.

# 🧠 4. Capacitatea AI de a învăța din date

🔍 Învățare automată a caracteristicilor: Modelele de tip Transformer (ex: Vision Transformer, Swin Transformer) pot extrage automat trăsături relevante din imagini, fără a necesita intervenție umană.

🔁 Transfer learning:

Pre-antrenare pe seturi mari (ex: ImageNet).

Fine-tuning pe seturi medicale specializate → obținerea unor performanțe excelente chiar și în contextul unor date limitate.



# 🔍 Analiza datelor de intrare

### 📁 Tipul de date

1. Datasetul utilizat este **Mini-MIAS (Mammographic Image Analysis Society Database)**, o colecție de **imagini mamografice digitale** utilizate în detecția cancerului mamar.

- Format imagini: `.pgm` (Portable GrayMap)
- Dimensiune: `1024 x 1024` pixeli
- Tip: **grayscale** (8-bit)
- Etichete asociate: stocate separat, cu informații despre:
  - Tipul țesutului mamar (Fatty, Glandular, Dense)
  - Prezența și tipul anomaliilor (circumscrise, calcificări, spiculate, etc.)
  - Severitate: **Benignă (B)** sau **Malignă (M)**

---

### 📊 Numărul de date

- Număr total imagini: **322**
- Număr pacienți: **161** (fiecare are câte 2 imagini – sân stâng și drept)

---

### 📈 Distribuția datelor

Distribuția este **dezechilibrată**, majoritatea imaginilor fiind normale (fără anomalie). Structura aproximativă este:

| Clasă          | Număr imagini | Procentaj |
|----------------|----------------|------------|
| **Normale**    | ~208           | ~65%       |
| **Benigne**    | ~63            | ~20%       |
| **Maligne**    | ~51            | ~15%       |
| **Total**      | 322            | 100%       |

> ℹ️ Anomaliile sunt împărțite în mai multe categorii (ex: `CIRC`, `CALC`, `SPIC`, `ARCH`, `ASYM`, `MISC`) și sunt etichetate corespunzător în fișierul de descriere.

---

### 🗂️ Observații

- Fiecare imagine are asociate coordonatele (x, y) ale centrului anomaliilor și un **raza** estimativă (doar pentru imagini anormale).
- Poate fi utilizat atât pentru **clasificare** (normal/benign/malign), cât și pentru **localizare/detecție** a leziunilor.


### 📁 Tipul de date

2. Datasetul utilizat este **DDSM (Digital Database for Screening Mammography)**, o colecție extensivă de mamografii digitale, creată pentru cercetare în domeniul detecției precoce a cancerului mamar.

- **Format imagini**: `LJPEG` (Lossless JPEG) – de obicei convertite în `.png`, `.jpg`, `.tiff` sau `.dcm` pentru procesare
- **Dimensiune**: variabilă, în general între `2000 x 3000` și `4000 x 6000` pixeli
- **Tip**: grayscale (`12-bit` inițial, convertit adesea la `8-bit`)
- **Etichete asociate**:
  - Tipul țesutului mamar: **Fatty**, **Glandular**, **Dense**
  - Tipul anomaliilor: **calcificări**, **mase**, etc.
  - Severitate: **Benignă (B)** sau **Malignă (M)**
  - Scor **BI-RADS**: între `0` și `5`

---

### 📊 Numărul de date

- **Număr total imagini**: ~**2.620** mamografii complet etichetate
- **Număr cazuri (pacienți)**: ~**1.040**  
  (fiecare caz include 2 sau 4 imagini – sân stâng/drept, în proiecțiile CC și MLO)

---

### 📈 Distribuția datelor

Distribuția este dezechilibrată, dar conține toate cele trei clase relevante:

| Clasă       | Număr imagini | Procentaj estimat |
|-------------|----------------|-------------------|
| **Normale** | ~1.100         | ~42%              |
| **Benigne** | ~800           | ~30%              |
| **Maligne** | ~720           | ~28%              |
| **Total**   | ~2.620         | 100%              |

> ℹ️ Anomaliile sunt etichetate în funcție de tip (ex. `mass`, `calcification`) și localizate cu precizie în imagine.

---

### 🗂️ Observații

- Fiecare imagine anormală include:
  - **măști de segmentare** (în formă binară)
  - **coordonate exacte** (x, y) și dimensiuni ale leziunii
  - **contururi segmentate** ale anomaliilor
- Suportă diverse sarcini de învățare automată:
  - 🟢 **Clasificare** (normal/benign/malign)
  - 🟡 **Detectare obiecte** (cu bounding box-uri)
  - 🔵 **Segmentare semantică** (mască exactă a leziunii)
- Unele cazuri conțin **multiple anomalii în aceeași imagine**

## 🔍 Analiza datelor de intrare

### 📁 Tipul de date

3. Datasetul utilizat este **INbreast 2012**, o colecție de mamografii digitale full-field, creată pentru a susține cercetarea în sisteme de detecție automată și diagnostic asistat (CAD) pentru cancerul mamar.

- **Format imagini**: `DICOM` (standard medical radiologic)
- **Dimensiune**: variabilă, dar toate imaginile sunt de înaltă rezoluție
- **Tip**: grayscale, capturate cu echipament **digital direct** (nu scanate)
- **Etichete asociate**:
  - Leziuni de tip: **mase**, **calcificări**, **asimetrii**, **distorsiuni**
  - **Contururi exacte** ale leziunilor (format `.xml`)
  - **Fișiere ROI** și **fișe medicale** corespunzătoare fiecărui caz
  - **Scor BI-RADS**, tip leziune, poziție, dimensiuni – disponibile în fișiere `.xls` și `.csv`

---

### 📊 Numărul de date

- **Număr total imagini**: **410** mamografii
- **Număr cazuri (pacienți)**: **115**
  - 90 cazuri cu ambele sâni (4 imagini per caz)
  - 25 cazuri după mastectomie (2 imagini per caz)
- **Volum total arhivă**: ~**9.01 GB**

---

### 📈 Distribuția datelor

Datasetul include atât cazuri **normale**, cât și **cu leziuni benigne sau maligne**. Distribuția aproximativă este:

| Clasă       | Număr cazuri (estimat) |
|-------------|-------------------------|
| **Benigne** | ~90                     |
| **Maligne** | ~116                    |
| **Normale** | restul (imaginile fără leziuni) |

> ℹ️ Unele imagini conțin **mai multe leziuni** adnotate – fiecare cu propriile contururi și descrieri.

---

### 🗂️ Observații

- Fiecare imagine are adnotări precise oferite de specialiști în **format XML**.
- Structura folderului include:
  - `AllDICOMs` – imagini DICOM
  - `AllROI` – fișiere cu zone de interes (Region of Interest)
  - `AllXML` – contururi ale leziunilor
  - `MedicalReports` – observații clinice și scoruri BI-RADS
- Sarcini AI posibile:
  - 🟢 **Clasificare** (normal/benign/malign)
  - 🟡 **Detecție obiecte** (localizarea leziunilor)
  - 🔵 **Segmentare** (mască exactă contur leziune)

---

### 📁 Tipul de date

4. Datasetul utilizat este **BCS-DBT (Breast Cancer Screening – Digital Breast Tomosynthesis)**, o colecție masivă de mamografii tridimensionale (DBT), destinată cercetării în detectarea automată a cancerului mamar.

- **Format imagini**: `DICOM` – standard medical pentru imagini radiologice
- **Dimensiune**: variabilă, în funcție de aparat; tipic 3D (volume cu sute de slice-uri per sân)
- **Tip**: grayscale, de înaltă rezoluție (de obicei 12-bit sau 16-bit)
- **Etichete asociate**:
  - Clasificare caz: `Normal`, `Actionabil`, `Benign` (biopsie confirmată), `Malign` (biopsie confirmată)
  - Anotări de tip **bounding box** pentru mase și distorsiuni arhitecturale
  - Coordonate 3D și fișiere CSV cu pozițiile leziunilor
  - Metadate BI-RADS și alte informații clinice

> ⚠️ Unele imagini necesită corectarea orientării pentru ca anotările să fie corecte. Funcții Python pentru procesare corectă sunt disponibile pe [GitHub](https://github.com/MaciejMazurowski/duke-dbt-data).

---

### 📊 Numărul de date

- **Total pacienți**: **5.060**
- **Total imagini (DICOM slices)**: **19.148**
- **Volum total**: ~**1.63 TB** de date
- **Distribuție pe seturi**:
  - **Training**: 4.362 cazuri
  - **Validation**: 280 cazuri
  - **Test**: 418 cazuri

---

### 📈 Distribuția datelor

Distribuția este ușor dezechilibrată, dar bine acoperită pentru toate clasele clinice majore:

| Clasă             | Exemple incluse               |
|------------------|-------------------------------|
| **Normal**       | fără leziuni detectabile      |
| **Benign**       | mase benigne confirmate       |
| **Malign**       | mase maligne confirmate       |
| **Actionabil**   | necesită evaluare suplimentară |

> 📊 Numărul exact de cazuri per clasă este disponibil în fișierele `.csv` incluse (ex: `group_classification.csv`), corespunzător cu articolul științific asociat.

---

### 🗂️ Observații

- Fiecare caz conține:
  - **imagini DBT în format DICOM**, în proiecții CC și MLO pentru ambii sâni
  - **fișiere CSV** cu:
    - clasificarea cazului
    - locația leziunilor (bounding box-uri)
    - path-uri pentru studii și imagini
- **Scenarii posibile de utilizare**:
  - 🟢 **Clasificare** (normal/benign/malign/actionabil)
  - 🟡 **Detecție de leziuni** (bounding box)
  - 🔵 **Segmentare volumetrică 3D** *(posibil în versiuni viitoare)*

---


https://github.com/user-attachments/assets/fb109c42-1e56-48a4-ac80-7c72915d8ada








https://github.com/user-attachments/assets/51f06137-a295-4ea9-b3f7-bfd2b6163026


