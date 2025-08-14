ps -ef | grep test.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep test.sh | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep test_on_1.sh | grep -v grep | awk '{print $2}' | xargs kill -9