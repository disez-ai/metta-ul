!(import! &self metta_ul:pdm)


!(pdm.read_csv pdm_test.csv (usecols (longitude latitude median_house_value)))
