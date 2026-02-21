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

Hugging Face Repository URL: https://huggingface.co/datasets/rushilara/h2-video-detections

Student Report:

1. Detector choice and configuration



I went with RF-DETR for this project. From what I could find online, it was doing really well on object detection (and segmentation) benchmarks, and I had enough compute to fine-tune it rather than use something smaller. So I used the medium segmentation variant (RFDETRSegMedium) and transfer-learned it on the carparts dataset so it would know the same part classes we care about for matching later.


RF-DETR is a DETR-style model with a DINOv2 backbone and a segmentation head, so it gives you boxes and masks. For my pipeline, I only needed the bounding boxes, class labels, and confidence scores. The matching step is all at the class level—I didn’t use the actual mask outputs for retrieval.



I trained on carparts-seg, which has 23 car-part classes (hood, front_bumper, wheel, mirrors, etc.). I used the standard train/val/test splits and ran for up to 100 epochs, early-stopping on validation mAP. I also used a lower learning rate for the encoder (1e-5) than for the decoder (1e-4), so the pretrained features wouldn’t be wiped out. I saved checkpoints every 10 epochs, and for inference, I used the best one by mAP (checkpoint_best_total.pth) for both the video and the query images. In inference, I only kept detections with a confidence threshold of 0.5 to keep the index from getting too noisy.



2. Video sampling strategy


I needed a per-frame index of which car parts appear in the video so I could later find contiguous time ranges where each part appears. I sampled the video at one frame per second. So frame 1 = 1 second, frame 2 = 2 seconds, and so on. Going faster (e.g., several frames per second) would have yielded nicer segment boundaries but would have cost more in storage and inference. Going slower would have been cheaper, but I could have missed short appearances. One fps felt like a reasonable middle ground.


For each of those frames, I ran my trained model once and stored every detection above the threshold: class label, bounding box (x_min, y_min, x_max, y_max), and confidence score. Then I grouped those by frame and video ID into a table, with each row representing a frame and containing a list of all detections for that frame. That table is what I used as the video index, and I exported it as video_detections.parquet.


3. Image-to-video matching logic


The idea is: given a query image (e.g., a RAV4 exterior shot), find a contiguous clip in the video where the same kinds of car parts show up.


First, I run the same detector on the query image and get the set of part classes it detects (hood, bumper, etc.). I don’t compare appearance or geometry between the query and the video—just which class labels are present. So it’s class-based matching, not instance-level or embedding-based.


From the video index I precomputed, for each class, the contiguous segments: runs of consecutive seconds (frame indices) where that class appears at least once. So for “hood,” I might have segments like seconds 10–25, 40–50, etc. For each query, I take its set of detected classes and gather all segments that involve any of those classes (union over the query’s classes). That gives me a bunch of time intervals where at least one part of the query is in the video. Then I pick the longest of those segments by duration, and that’s the clip I return for that query. I store the query image id, start_sec, end_sec, and a YouTube embed URL with those times so you can click through and watch that clip.


A few design choices: I only do class-level matching (no embeddings or re-id). I don’t require all query parts to appear in the same segment—any segment where any of the parts appear counts. And I only return one clip per query (the longest) to keep things simple.


4. Failure cases and limitations


A few things went wrong, or I felt limited.


Some query images didn’t return any detections above the threshold (or only 1 or 2). For those, there’s nothing to match, so the pipeline correctly gives them a null clip—no start/end time, no URL. So when a query “isn’t in the video,” it often just means the detector didn’t see any of the 23 parts in that image. That can be because of angle, occlusion, or because the query images (e.g., RAV4 exteriors) are from a different distribution than the carparts training data. The threshold also matters: if I’d set it higher, I’d have fewer false positives but more of these “no match” cases.


Another issue was that many different query images pointed to the same clip. I think that’s partly because many queries detect the same subset of parts (e.g., hood, front bumper, front glass). If the video has a single continuous stretch where those parts are visible, that stretch wins as the “longest” for every such query. So you get many-to-one: lots of queries → same clip. The class labels are also pretty coarse (just “hood” or “bumper,” not which car or which instance), and we only return one clip per query, so there’s not much to differentiate them. Doing something with embeddings or returning multiple/ranked clips could help.


The model is also worse on small parts (lights, mirrors) than on big ones (hood, doors, glass), so queries that are mostly small parts sometimes get no or noisy detections and wrong or missing clips. The carparts dataset has an “object” class that was basically unused. And the 0.5 threshold is fixed—changing it would trade off precision vs “no match” and vs noisier segments. Finally, the pipeline is set up for one video; extending to multiple videos would need video_id in the index and in the URLs.


So in short: the main limitations are (1) some queries don’t get detections and correctly get no clip, (2) many queries map to the same clip because of class overlap and the “longest segment” rule, (3) model/dataset limits (small parts, label set, threshold), and (4) matching is only by class, not by appearance or instance. I’d improve it by trying embedding-based retrieval, returning more than one clip per query, or tuning the threshold and how we pick segments.




