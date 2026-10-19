# Execution Guide: do these in order

Tick each box. Take the screenshots marked 📸 and save them in `screenshots/` with the suggested file names.

## Phase 0: Prerequisites (one time)

- [ ] Python 3.11 or newer: `python --version`
- [ ] Git: `git --version`
- [ ] A GitHub account
- [ ] **Docker Desktop** installed and running (Windows: enable WSL 2 when asked)
- [ ] Docker Desktop → Settings → Kubernetes → **Enable Kubernetes** (takes a few minutes). This gives you `kubectl` and a local cluster, so no cloud account is needed.
- [ ] Verify: `docker --version`, `kubectl version --client`

## Phase 1: Put the project in place

1. Unzip the project. Open the folder in VS Code.
2. Copy your own `data/raw` and `data/processed` files over only if you want; the scripts regenerate them.
3. Terminal (PowerShell) in the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Phase 2: Data, EDA, training

```powershell
python -m src.data.download_data
python -m src.data.preprocess
```

- [ ] Open `notebooks/01_eda.ipynb` (kernel: `.venv`), **Run All**, write your observations. 📸 `eda_correlation.png`
- [ ] Train:

```powershell
python -m src.models.train
```

This creates `models/best_model.joblib`, `models/model_metadata.json`, `reports/model_comparison.csv` and plots, and logs everything to MLflow.

## Phase 3: MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000, click the experiment **heart-disease-classification**.
- 📸 `mlflow_runs.png`: the runs table
- Tick the two main runs (`logistic_regression`, `random_forest`) → **Compare**. 📸 `mlflow_compare.png`
- Open one run → Artifacts tab. 📸 `mlflow_artifacts.png`

Stop the server with Ctrl+C.

## Phase 4: Tests and lint

```powershell
flake8 .
pytest --cov=src
```
📸 `pytest_pass.png`

## Phase 5: Pin your versions (reproducibility, important for Docker)

The model is saved with a pickle-style format, so the library versions that **load** it must match those that **trained** it.

```powershell
pip freeze | findstr /i "scikit-learn numpy pandas joblib"
```

Open `requirements-api.txt` and replace the ranges for those four packages with the exact versions, for example `scikit-learn==1.5.2`. Do the same in `requirements.txt` if you want a fully locked file. You can also save a complete lock with `pip freeze > requirements-lock.txt`.

Then **retrain once more** so everything matches: `python -m src.models.train`.

## Phase 6: Run the API locally

```powershell
uvicorn src.api.app:app --reload
```

- Browser: http://localhost:8000/docs → try `/predict`. 📸 `api_docs.png`
- PowerShell test (note: use `Invoke-RestMethod`, because `curl` in PowerShell is an alias with different syntax):

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile scripts/sample_request.json
```

Or use `curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@scripts/sample_request.json"`. 📸 `api_predict.png`

## Phase 7: GitHub and CI/CD

1. On github.com → **New repository** → name `heart-disease-mlops`, public, no README.
2. In the project folder:

```powershell
git init
git add .
git commit -m "feat: end-to-end MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/heart-disease-mlops.git
git push -u origin main
```

(If asked for a password, GitHub needs a **personal access token**: GitHub → Settings → Developer settings → Tokens, or sign in through the browser popup.)

3. Repo → **Actions** tab. The workflow *CI/CD Pipeline* runs automatically: lint → test → train → docker. 📸 `actions_success.png` (the green graph) and 📸 `actions_artifacts.png` (artifacts at the bottom of a run).
4. **Prove the pipeline fails on errors** (a requirement): edit any test to `assert False`, commit, push, see the red failure and the log. 📸 `actions_failure.png`. Then revert with `git revert HEAD` and push.

`models/best_model.joblib` is committed on purpose so Docker can build locally. (In big projects you would store models in a registry or object storage instead.)

## Phase 8: Docker

```powershell
docker build -t heart-disease-api:1.0 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0
docker ps
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile scripts/sample_request.json
docker logs heart-api
```
📸 `docker_build.png`, `docker_run_predict.png`, `docker_logs.png`

Stop it: `docker stop heart-api; docker rm heart-api`

## Phase 9: Kubernetes (local)

The image `heart-disease-api:1.0` must exist locally (Phase 8). With Docker Desktop Kubernetes it is shared automatically.

```powershell
kubectl config use-context docker-desktop
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods
kubectl get svc heart-disease-api
```

Wait until both pods are `Running` and `READY 1/1`. The service `EXTERNAL-IP` shows `localhost` on Docker Desktop.

```powershell
Invoke-RestMethod -Uri http://localhost/health
Invoke-RestMethod -Uri http://localhost/predict -Method Post -ContentType "application/json" -InFile scripts/sample_request.json
kubectl logs deployment/heart-disease-api
```
📸 `k8s_pods_svc.png`, `k8s_predict.png`, `k8s_logs.png`

**Minikube instead:** `minikube start`, then `minikube image load heart-disease-api:1.0`, apply the manifests, and in a second terminal run `minikube tunnel` to get an external IP.

Troubleshooting: `ImagePullBackOff` means the image isn't local (rebuild/load it). `CrashLoopBackOff` means check `kubectl logs <pod>` (usually a missing `models/best_model.joblib` at build time).

Clean up when finished: `kubectl delete -f k8s/`

## Phase 10: Monitoring (Prometheus + Grafana)

Stop the kubectl service and other containers using port 8000/80 first, then:

```powershell
docker compose up --build -d
```

1. Generate traffic (run several times, plus one bad request):

```powershell
1..30 | ForEach-Object { Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile scripts/sample_request.json | Out-Null }
```

2. Prometheus http://localhost:9090 → Status → Targets shows `heart-disease-api` UP. 📸 `prometheus_targets.png`. Query `api_requests_total`. 📸 `prometheus_query.png`
3. Grafana http://localhost:3000 (admin / admin) → Dashboards → **Heart Disease API**. 📸 `grafana_dashboard.png`
4. Logs: `docker compose logs api`. 📸 `api_request_logs.png`

Stop: `docker compose down`

## Phase 11: Video (a few minutes)

Record your screen (Windows: Win+G, or OBS) and show, in this order: repo on GitHub → Actions green run → `mlflow ui` comparison → `docker build` and `docker run` → `/predict` call → `kubectl get pods` and prediction through the service → Grafana dashboard.

## Phase 12: Report

Fill in `docs/REPORT_TEMPLATE.md` with your real numbers (from `reports/model_comparison.csv` and MLflow), paste in the screenshots, then export to PDF (VS Code extension *Markdown PDF*, or paste into Word). Target about 10 pages. If you send me your metrics table and screenshots, I can generate the finished report file for you.

## Final submission checklist

- [ ] GitHub repo public, all folders present, Actions green
- [ ] `screenshots/` populated
- [ ] Report PDF/DOCX (about 10 pages) with repo link
- [ ] Video recorded
- [ ] README has repo link and access instructions
