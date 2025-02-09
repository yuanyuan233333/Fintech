import joblib


def save(model, path):
    if 'keras' in str(type(model)):
        model.save(path)
    else:
        joblib.dump(model, path)
