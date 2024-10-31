# test.py
def evaluate_model(model, test_images, test_labels):
    results = model.evaluate(test_images, test_labels)
    with open("test.dat", 'w') as f:
        f.write(" ".join(map(str, results)))
    return results

