import yaml
import json


def convert_cntl_to_yamljson(cntl_file, out_file):
    config = {}
    with open(cntl_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            key, value = line.split("=")
            value = value.strip()
            if value.lower() in ("yes", "no"):
                value = {"yes": True, "no": False}[value.lower()]
            elif "," in value:
                if value.startswith('"'):
                    value = value[1:]
                if value.endswith('"'):
                    value = value[:-1]
                value = value.split(",")
                try:
                    value = [int(v.strip()) for v in value if len(v.strip()) > 0]
                except ValueError:
                    try:
                        value = [float(v.strip()) for v in value if len(v.strip()) > 0]
                    except ValueError:
                        pass
            else:
                try:
                    value = int(value.strip())
                except ValueError:
                    try:
                        value = float(value.strip())
                    except ValueError:
                        pass
            if key.strip().upper() in (
                "WALL_TIME",
                "CYCLE_TIME",
                "CHECKPOINT_TIME",
                "SUBJOBS_BUFFER_SIZE",
                "RE_SETUP",
                "JOB_TRANSPORT",
            ):
                continue
            config[key.strip()] = value

    with open(out_file, "w") as f:
        if out_file.endswith(".yaml"):
            yaml.dump(config, f, default_flow_style=None, sort_keys=False)
        elif out_file.endswith(".json"):
            json.dump(config, f)


if __name__ == "__main__":
    import sys

    convert_cntl_to_yamljson(sys.argv[1], sys.argv[2])
