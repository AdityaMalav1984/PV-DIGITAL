.PHONY: setup test run dashboard clean
setup:
	pip install -r requirements.txt
test:
	pytest -q tests/
run:
	python run_pipeline.py
dashboard:
	streamlit run app.py
clean:
	rm -rf outputs/*.csv outputs/*.csv.gz outputs/tables/*.csv outputs/figures/*.png __pycache__ src/__pycache__ tests/__pycache__
