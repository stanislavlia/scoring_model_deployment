echo "Stress tests using batchmode; size=128"
echo "============================================"
ab -n 128 -c 1 -p batch128.json -T application/json -rk http://localhost:8081/prediction   
echo "DONE"

echo "============================================"
echo "Stress tests using batchmode; size=256"
ab -n 128 -c 1 -p batch256.json -T application/json -rk http://localhost:8081/prediction   
echo "DONE"
echo "============================================"
echo "Stress tests using batchmode; size=1024"
ab -n 128 -c 1 -p batch1024.json -T application/json -rk http://localhost:8081/prediction   
echo "DONE"