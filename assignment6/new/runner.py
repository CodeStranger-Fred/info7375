image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime")
    .pip_install(["transformers==4.57.1", "colorama"])
    .add_local_file(local_path="entry.py", remote_path="/root/entry.py")
)

app = modal.App(name=Path.cwd().name)

@app.function(gpu="H100", image=image, timeout=24 * 60 * 60)
def run():
    from entry import main
    main()

