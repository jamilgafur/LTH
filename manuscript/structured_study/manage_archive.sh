#!/bin/bash

# Usage:
# ./manage_archive.sh break    # to archive + split
# ./manage_archive.sh combine  # to recombine + optionally extract

DIR="pruning_checkpoints"
ARCHIVE="${DIR}.tar.xz"
PART_PREFIX="${DIR}.part_"
CHUNK_SIZE="90M"

if [ $# -ne 1 ]; then
    echo "Usage: $0 [break|combine]"
    exit 1
fi

ACTION=$1

case "$ACTION" in
    break)
        if [ ! -d "$DIR" ]; then
            echo "Error: directory $DIR not found!"
            exit 1
        fi
        echo "Creating archive $ARCHIVE from $DIR..."
        tar -cJvf "$ARCHIVE" "$DIR"

        echo "Splitting $ARCHIVE into $CHUNK_SIZE chunks..."
        split -b "$CHUNK_SIZE" "$ARCHIVE" "$PART_PREFIX"

        echo "Done. Files created:"
        ls ${PART_PREFIX}*

        echo "Optional: removing original archive to save space..."
        # Uncomment the next line if you want to delete the full archive
        # rm "$ARCHIVE"
        ;;

    combine)
        PARTS=$(ls ${PART_PREFIX}* 2>/dev/null)
        if [ -z "$PARTS" ]; then
            echo "Error: No parts found with prefix $PART_PREFIX"
            exit 1
        fi

        echo "Combining parts into $ARCHIVE..."
        cat ${PART_PREFIX}* > "$ARCHIVE"

        echo "Done. You can now extract the archive using:"
        echo "tar -xJvf $ARCHIVE"
        ;;

    *)
        echo "Invalid option: $ACTION"
        echo "Usage: $0 [break|combine]"
        exit 1
        ;;
esac
