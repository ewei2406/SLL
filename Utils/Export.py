import csv
from datetime import date
from os.path import exists
import json

def makeFile(filename, template='none'):
    f = open(filename, 'x')
    if template == 'json':
        with open(filename, 'a') as f:
            f.write("{}")


def checkIfFileExists(filename):
    return exists(filename)


def getCsvHeader(filename):
    colNames = []
    with open(filename, 'r') as csv_file:
        for row in csv_file:
            colNames = row
            break
    return colNames.replace("\n", "").split(",")


def appendCsv(filename, string):
    with open(filename, 'a') as csv_file:
        csv_file.write("\n" + ",".join(string))


def setCsvHeader(filename, header):
    with open(filename, 'w') as csv_file:
        csv_file.write(",".join(header))


def saveData(filename: str, data: dict) -> None:
    """
    Apprends a dictionary of [column: value] to csv specified by filename
    if not csv exists, create the csv with column names
    Will fail if csv exists but has mismatching column names
    """
    if not checkIfFileExists(filename):
        makeFile(filename)
        setCsvHeader(filename, [k for k in data])
    
    csvHeader = getCsvHeader(filename)

    missingCol = []
    for column in csvHeader:
        if column not in data:
            missingCol.append(column)
    if len(missingCol) > 0:
        print("Error: Column mismatch (" + ",".join([str(k) for k in missingCol]) + " in csv but not in data to append)")
        return False
    
    missingCol = []
    for column in data:
        if column not in csvHeader:
            missingCol.append(column)
    if len(missingCol) > 0:
        print("Error: Column mismatch (" + ",".join([str(k) for k in missingCol]) + " in data to append but not in csv)")
        return False

    appendCsv(filename, [str(data[k]) for k in csvHeader])


def save_var(key: str, value: str, filename="./temp.json"):
    """
    Saves a variable into the temp file if it exists, or creates a new one
    """
    if (not exists(filename)):
        print("Creating temp file...")
        makeFile(filename, template='json')
    with open(filename, 'r') as read:
        print(f"Saving data for [{key}]")
        data = json.load(read)
        if key in data:
            print(f"Warning: overwriting existing temp data for [{key}]")
        data[key] = value
        with open(filename, 'w') as write:
            write.write(json.dumps(data))


def load_var(key: str, filename="./temp.json"):
    """
    Gets a variable out of temp file if it exists, or None otherwise
    """
    if exists(filename):
        print(f"Loading data for [{key}]")
        with open(filename, 'r') as f:
            data = json.load(f)
            if key in data:
                return data[key]
    print(f"Failed to find data for [{key}]")
    return None


if __name__ == "__main__":
    d = load_var("cora")
    if d != None:
        print(d)
    else:
        save_var("cora", [1, 2, 3])

