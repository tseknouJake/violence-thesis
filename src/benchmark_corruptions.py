import argparse

from eval import make_loader, evaluate
from models import make_model
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/bus-violence")
    parser.add_argument("--split-file", default="splits/bus_violence_splits.json")
    parser.add_argument("--model", default="r3d_18")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["downsample", "compression", "low_light", "gaussian_blur", "motion_blur", "occlusion", "camera_shake", "fps_drop"],
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(name=args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)

    header = "corruption,severity,acc,precision,recall,f1,auc,cm"
    print(header)

    for corruption in args.corruptions:
        for severity in [1, 2, 3]:
            args.corruption = corruption
            args.severity = severity
            loader = make_loader(args, "test")
            acc, p, r, f1, auc, cm = evaluate(model, loader, device)
            print(f"{corruption},{severity},{acc:.3f},{p:.3f},{r:.3f},{f1:.3f},{auc:.3f},{cm}")


if __name__ == "__main__":
    main()
