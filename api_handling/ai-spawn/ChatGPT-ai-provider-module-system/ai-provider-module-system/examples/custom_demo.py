from ai_provider_system import AIModel, setup_custom


def summarize_locally(data, instruction):
    # Demonstrates the callback contract; replace with your own model invocation.
    return f"Demo received {len(data)} characters. Requested task: {instruction}"


if __name__ == "__main__":
    model = AIModel(setup_custom(summarize_locally))
    print(model.summarize({"sales": [10, 20, 30]}, "Describe sales."))
