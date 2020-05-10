init:
	pip2 install -r requirements.txt

test:
	nosetests tests

distr:
	python2 setup.py bdist_wheel
	twine upload dist/*

clean:
	rm *.pyc
