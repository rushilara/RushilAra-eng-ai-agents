# Video Detections and Query Clips

Parquet outputs for the video detection and image semantic search pipeline (car-parts detector on video + RAV4 query images).

## Files

- **video_detections.parquet** — One row per video frame; each row has a list of object detections for that frame.
- **query_longest_clips.parquet** — One row per query image; each row has the longest contiguous video clip where the detected car parts appear, with a YouTube embed URL.

## Schema

### video_detections.parquet

| Column         | Type   | Description                                                                 |
|----------------|--------|-----------------------------------------------------------------------------|
| `video_id`     | string | Identifier of the video (e.g. `"input_video"`).                             |
| `frame_index`  | int    | Frame number (1-based).                                                    |
| `detections`   | list   | List of detections for this frame. Each element has the fields below.      |

Each element in `detections`:

| Field              | Type   | Description                                              |
|--------------------|--------|----------------------------------------------------------|
| `class_label`      | string | Car part class (e.g. `"hood"`, `"front_bumper"`, `"wheel"`). |
| `bounding_box`     | list   | `[x_min, y_min, x_max, y_max]` in image coordinates.    |
| `confidence_score` | float  | Detection confidence in [0, 1].                         |

### query_longest_clips.parquet

| Column              | Type   | Description                                                                 |
|---------------------|--------|-----------------------------------------------------------------------------|
| `query_image_id`    | int    | Index of the query image in the dataset.                                   |
| `start_sec`         | int    | Start time of the longest contiguous clip (seconds).                      |
| `end_sec`           | int    | End time of the longest contiguous clip (seconds).                       |
| `youtube_embed_url` | string | URL to embed the clip (e.g. `https://www.youtube.com/embed/VIDEO_ID?start=...&end=...`). |
