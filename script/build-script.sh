#!/usr/bin/env bash
set -x
echo "Changes:"
cat $1

# TeamCity style
CHANGED_MODEL_FILES="$(cat $1 | \
  grep -v REMOVED | \
  cut -d \: -f 1 | \
  grep -e model | \
  grep .py | \
  grep -v util | \
  grep -v baseline | \
  grep -v png \
  )"
echo ${CHANGED_MODEL_FILES}

# Comment out line below to enable automated script execution
#CHANGED_MODEL_FILES="echo ${CHANGED_MODEL_FILES} | grep DISABLE_AUTOMATED_EXECUTION"

set -e
cd model
for FILE in ${CHANGED_MODEL_FILES}
do
	python3 ../${FILE}
done

echo "built!"