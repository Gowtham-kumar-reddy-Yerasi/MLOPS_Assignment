# Heart Disease Risk Prediction: MLOps Report

**Name / ID:** <<  >>  **Repository:** <<GitHub URL>>  **Date:** <<  >>

## 1. Introduction
Problem statement, why heart disease prediction matters, and the goal: a reproducible, deployed, monitored classifier. (about 1/2 page)

## 2. Setup and Installation
Prerequisites, virtual environment, `pip install -r requirements.txt`, command sequence from the README (download, preprocess, train, test, serve), Docker and Kubernetes commands. Mention the UCI fallback.

## 3. Data Acquisition and EDA
- Source: UCI Heart Disease (Cleveland), 303 patients, 13 features, binary target (0 vs 1-4 collapsed).
- Cleaning: column names added, `?` treated as missing, target binarised, duplicates removed.
- **Design decision:** imputation deferred to the pipeline to avoid data leakage.
- Insert figures from `reports/figures/`: class balance, histograms, boxplots, categorical vs target, correlation heatmap. One paragraph of interpretation each.
- Summary of findings: <<your bullets>>

## 4. Feature Engineering and Modelling
- Pipeline diagram: median impute + StandardScaler for numeric; most-frequent impute + one-hot for categorical.
- Models: Logistic Regression and Random Forest, with reasons.
- Tuning: GridSearchCV, 5-fold stratified CV, ROC-AUC objective; grids used.
- Results table (paste `reports/model_comparison.csv`): accuracy, precision, recall, ROC-AUC (CV mean ± std, and test).
- Model selection justification (chosen by CV, not test). Discussion of recall in a medical setting, test set size (about 61 patients) and limitations.
- Insert ROC curves and confusion matrix.

## 5. Experiment Tracking (MLflow)
What is logged (params, metrics, plots, model, nested runs per grid candidate). Screenshots: runs table, comparison view, artifacts. Summary of what MLflow showed.

## 6. Model Packaging and Reproducibility
Saved pipeline (joblib) containing preprocessing + model, `model_metadata.json`, pinned requirements, seeds (`random_state=42`), same-version rule for loading.

## 7. CI/CD and Automated Testing
Tests written (list them, what each checks), coverage, workflow stages (lint → test → train → docker build and smoke test), artifacts uploaded. Screenshots: green run, artifacts, and the deliberately failing run proving the pipeline stops on errors.

## 8. Containerisation
Dockerfile explanation (slim base, dependency layer caching, non-root user, healthcheck). Screenshots: build, run, `/predict` response. Sample request and response JSON.

## 9. Deployment
Kubernetes Deployment (2 replicas, probes, resource limits) and LoadBalancer Service. Screenshots: `kubectl get pods,svc`, successful prediction, logs. Access instructions / URL.

## 10. Monitoring and Logging
Request logging (method, path, status, latency; no patient data logged), Prometheus metrics exposed at `/metrics`, Grafana dashboard panels. Screenshots.

## 11. Architecture Diagram
Paste the diagram from README (render the Mermaid or redraw it).

## 12. Conclusion and Future Work
Results summary, lessons learned, improvements: threshold tuning to raise recall, nested CV, model registry, data and drift monitoring, automated retraining, cloud deployment (EKS/GKE/AKS), Helm chart.

## References
UCI repository, scikit-learn, MLflow, FastAPI, Prometheus and Grafana docs.
