echo "Stress tests using batchmode; size=128"
ab -n 128 -c 3 -p batch128.json -T application/json -rk http://localhost:81/prediction   
echo "DONE"


echo "Stress tests using batchmode; size=256"
ab -n 128 -c 3 -p batch256.json -T application/json -rk http://localhost:81/prediction   
echo "DONE"