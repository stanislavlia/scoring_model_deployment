# homecredit_scoring
This is the end-to-end project of creating a credit scoring model for home credit bank 


Server Monitoring with Prometheus and Grafana

This configuration establishes a robust monitoring solution for our server using Prometheus for data collection and Grafana for data visualization. Our setup is defined and managed through Docker Compose, ensuring easy deployment and scalability.

Components:

  1) Cred Scoring Service: Our primary application service, cred_scoring, is containerized and exposed on port 8081. This service is the focus of our monitoring efforts.

  2) Prometheus: As our monitoring backbone, Prometheus is configured to scrape and store metrics from our cred_scoring service. It is set up to dynamically collect metrics, providing real-time insights into the performance and health of our application. Prometheus data is persistently stored in a Docker volume (prometheus-data), ensuring that our historical data is preserved across container restarts.

  3) Grafana: For effective data visualization, we use Grafana. It is connected to Prometheus as the data source, allowing us to create comprehensive dashboards that offer a clear view of our server's performance metrics. Grafana is also configured with a persistent volume (grafana-data), safeguarding our dashboard configurations and customizations.
## How to start
Run applications:
```bash
docker-compose up
```

This command will run 3 applications: cred_scoring(server with ML model), prometheus, grafana.


#### Default adresses:

  cred_scoring - http://localhost:8081
  
  prometheus  - http://localhost:9090
  
  grafana   - http://localhost:3000
  
  ## How to get predictions:
In order to get predictions, you need to send a particular set of features (listed in expected_feature.txt) to http://localhost:8081/prediction.

See an example of JSON of features [here](stress_test/batch1.json) for batch of size 1.

See an example of JSON of features [here](stress_test/batch8.json) for batch of size 8.


### Stress test
In order to collect statistics using Apache Benchmark, go to the directory called stress_test and execute shell script.
```bash
./tesh.sh
```
This test will check how our sever works when batch_size=128 and when batch_size=1024.


### Monitoring server using Grafa
After you ran all 3 apps using docker-compose, you can open http://localhost:3000 and login.
Use password = admin and login = admin. Once you logged in, you need to add our Prometheus server as first your data source.
After that, you can build a dashboard for server monitoring.
