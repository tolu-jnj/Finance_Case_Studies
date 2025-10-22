# Time Series Forecasting Portfolio Application

This repository contains a production-ready Streamlit application for time series forecasting, demonstrating both machine learning and classical approaches across two case studies.

## 🎯 Features

- Interactive time series visualization and exploration
- Multiple forecasting models comparison
- Data quality analysis and seasonal decomposition
- Model performance metrics and diagnostics
- AI-assisted development documentation

## 🛠️ Local Development Setup

1. Clone the repository
```bash
git clone <repository-url>
cd forecasting-app
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## 🐳 Docker Deployment

### Using Docker

1. Build the Docker image
```bash
docker build -t forecasting-app .
```

2. Run the container
```bash
docker run -p 8501:8501 forecasting-app
```

### Using Docker Compose

1. Start the application
```bash
docker-compose up -d
```

2. Stop the application
```bash
docker-compose down
```

## ☁️ Akash Network Deployment

1. Build and tag the Docker image
```bash
docker build -t your-dockerhub-username/forecasting-app:latest .
```

2. Push to Docker Hub
```bash
docker push your-dockerhub-username/forecasting-app:latest
```

3. Update deploy.yaml
- Replace `your-dockerhub-username` with your actual Docker Hub username
- Adjust resource requirements if needed

4. Deploy using Akash CLI
```bash
# Initialize Akash deployment
akash tx deployment create deploy.yaml --from <your-account> --chain-id akashnet-2 --node <node-address>

# View your deployments
akash query deployment list --owner <your-account-address>

# Accept bid and create lease
akash tx market lease create --from <your-account> --chain-id akashnet-2 --node <node-address>
```

## 🔧 Environment Variables

The application supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| STREAMLIT_SERVER_PORT | Server port | 8501 |
| STREAMLIT_SERVER_ADDRESS | Server address | 0.0.0.0 |
| STREAMLIT_SERVER_HEADLESS | Run in headless mode | true |

## 🚀 Production Deployment Checklist

- [ ] Update Docker Hub credentials
- [ ] Configure Streamlit secrets
- [ ] Set up monitoring
- [ ] Configure backup strategy
- [ ] Set up CI/CD pipeline

## 🔍 Troubleshooting

### Common Issues

1. Port already in use
```bash
# Check what's using port 8501
lsof -i :8501
# Kill the process
kill -9 <PID>
```

2. Docker container not starting
```bash
# Check container logs
docker logs forecasting-app
```

3. Data loading errors
- Verify data files exist in correct locations
- Check file permissions
- Validate CSV format and encoding

### Health Checks

The application includes built-in health checks:
- Docker container: http://localhost:8501/_stcore/health
- Akash deployment: Automated health checks every 30s

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Akash Network Documentation](https://docs.akash.network/)

## 📝 License

MIT License. See LICENSE file for details.