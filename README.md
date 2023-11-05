# Creating and Validating Synthetic Patient Healthcare Trajectories

## Abstract
A patient healthcare trajectory is the sequence of events in a patient's healthcare journey, such as diagnoses, treatments, hospitalizations, prescribed medications, and outcomes. They assist healthcare providers in understanding the development and treatment of medical conditions, improving diagnoses and treatment plans. Research involving patient healthcare trajectories has yielded valuable insights that enhance the administration, diagnosis, and timely availability of treatment, improving clinical protocols. However, accessing and sharing medical data for research purposes is often a complex and time-consuming task due to the sensitive nature of patient medical information, which necessitates strict confidentiality measures. Synthetic data is a practical solution to the challenges in healthcare research related to data access, patient privacy, and data scarcity.

Previous studies on Generative adversarial networks (GANs) and medical time series data have focused on patient vital signs, such as heart rate and blood pressure, which are continuous and frequent. These datasets focus on a short window of time the first 24 to 48 hours, and remove or augment records with missing values. This approach neglects the discrete and rare medical events that play a pivotal role in clinical decision-making and the often incomplete nature of medical data. Furthermore, a substantial challenge arises from the rigidity of most GAN synthesisers, which requires uniformity in the input matrix dimensions. Typically, patient trajectories include a unique timestamp associated with each event occurrence, and the number of events can vary widely from patient to patient. However, most GAN models used for trajectory synthesis require uniformity in the dimensions of the input matrix.

My thesis addresses gaps in previous research by focusing on discrete and sparse time series data, specifically by looking at the occurrence of lab tests and medication administrations during a patient’s stay in the ICU. This paper introduces a data preprocessing pipeline for converting Electronic Health Record (EHR) data into patient healthcare trajectories, with a focus on preserving temporal coherence, managing missing data, and handling irregular time intervals. We provide a direct comparison of how a variety of GAN models perform against validation metrics measuring statistical properties, event correlations, data distribution and sequential fidelity. We found most synthetic models failed to capture the underlying characteristics of patient healthcare trajectories when faced with highly sparse data. This study underscores the pressing need to address the unique challenges posed by discrete time series medical data and develop more robust synthetic data model that can better serve the complex requirements of machine learning applications and research in healthcare.


## Models
TimeGAN: https://github.com/jsyoon0823/TimeGAN
ehrMGAN: https://github.com/jli0117/ehrMGAN
DeepEcho: https://github.com/sdv-dev/DeepEcho
CorGAN: https://github.com/astorfi/cor-gan
PATEGAN: https://github.com/BorealisAI/private-data-generation
DoppelGANger: https://github.com/fjxmlzn/DoppelGANger

 