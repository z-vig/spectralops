# pretty_print_runtime.py

def pretty_print_runtime(
    runtime: float,
    process_name: str = "Process"
) -> None:
    """
    Convenience function for printing run time in a nice way.

    Paramters
    ---------
    runtime: float
        Run time in seconds.
    """
    if runtime < 60:
        print(f"{process_name} complete in {runtime:.2f} seconds")
    elif runtime > 60 and runtime < 3600:
        print(f"{process_name} complete in {runtime // 60:.0f} minute(s)"
              f"{runtime % 60:.2f} seconds")
