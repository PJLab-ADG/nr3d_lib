# NOTE: Insert parent directory into PYTHONPATH
# Usage: source set_env.sh

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_dir=$(realpath $script_dir)
parent_dir=$(dirname $script_dir)
export PYTHONPATH="${parent_dir}":$PYTHONPATH
echo "Added $parent_dir to PYTHONPATH"