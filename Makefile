init:
	pip install -r requirements.txt

test:
	nosetests tests

clean:
	rm *.pyc
