from mlflow_lab.training import load_training_data


def test_data_split_is_deterministic() -> None:
    first_split = load_training_data(test_size=0.2, random_state=42)
    second_split = load_training_data(test_size=0.2, random_state=42)

    x_train_first, x_test_first, y_train_first, y_test_first = first_split
    x_train_second, x_test_second, y_train_second, y_test_second = second_split

    assert x_train_first.equals(x_train_second)
    assert x_test_first.equals(x_test_second)
    assert y_train_first.equals(y_train_second)
    assert y_test_first.equals(y_test_second)


def test_data_split_preserves_all_rows() -> None:
    x_train, x_test, y_train, y_test = load_training_data(
        test_size=0.2,
        random_state=42,
    )

    assert len(x_train) + len(x_test) == 569
    assert len(y_train) + len(y_test) == 569
