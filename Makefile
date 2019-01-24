init:
	pip install -r requirements.txt

test:
	nosetests tests

distr:
	python setup.py bdist_wheel
	twine upload dist/*

clean:
	rm *.pyc
