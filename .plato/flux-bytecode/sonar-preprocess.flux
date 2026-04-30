; SonarVision Flux Program — NMEA Sonar Preprocessing
; Converts raw NMEA sonar sentences into normalized sonar images
; for SonarVision encoder input.
;
; Flux runtime: flux-runtime (Python bytecode VM)
; Integration: called by sonar_vision/data/preprocessing.py::parse_nmea_sonar()

.section .data
    MAX_BEARING_BINS   128
    MAX_DEPTH_M        200
    DB_SILENCE        -80.0
    DB_NORMALIZE_MIN  -80.0
    DB_NORMALIZE_MAX    0.0

.section .text
    ; Entry: parse_nmea_sonar(nmea_string) → sonar_image[128][200]
    ; Input: NMEA string like "$PSDVS,15.2,45.0,-30.5,3.0*4A"
    ; Output: normalized float array (128 x 200)

    ; Step 1: Validate checksum
    LOAD r0, $input          ; raw NMEA string
    CALL validate_checksum   ; r1 = valid (0/1)
    JZ r1, @error            ; skip if invalid

    ; Step 2: Extract fields
    CALL extract_fields      ; r2 = depth, r3 = bearing, r4 = intensity, r5 = beam_width

    ; Step 3: Convert bearing to bin index
    ; bearing range: -90 to +90 degrees → 0 to 127
    LOAD r6, 90.0
    FADD r3, r6              ; shift to 0-180
    LOAD r7, 180.0
    FDIV r3, r7              ; normalize to 0-1
    LOAD r8, MAX_BEARING_BINS
    FMUL r3, r8              ; scale to 0-127
    ISTORE r3, bearing_idx

    ; Step 4: Convert depth to pixel index
    LOAD r9, MAX_DEPTH_M
    FDIV r2, r9              ; normalize to 0-1
    FMUL r2, r8              ; scale to 0-199
    ISTORE r2, depth_idx

    ; Step 5: Normalize intensity dB to [0, 1]
    LOAD r10, DB_SILENCE
    FSUB r4, r10             ; shift to 0-80
    LOAD r11, 80.0
    FDIV r4, r11             ; normalize to 0-1
    FSTORE r4, norm_intensity

    ; Step 6: Write to image buffer
    LOAD r12, $image_buffer
    IMUL r13, bearing_idx, MAX_DEPTH_M
    IADD r13, depth_idx
    STORE r12[r13], norm_intensity

    ; Step 7: Apply depth gradient channel
    ; gradient = |intensity[i][d] - intensity[i][d-1]|
    CALL compute_gradient    ; writes to channel 1

    ; Step 8: Apply depth normalization channel
    CALL depth_normalize     ; writes to channel 2

    ; Step 9: Accumulate returns channel
    CALL accumulate_returns  ; writes to channel 3

    RET $image_buffer        ; return 4-channel sonar image

error:
    LOAD r0, -1
    RET r0

.section .functions
    validate_checksum:
        ; XOR bytes between $ and *
        ; Compare to hex checksum after *
        RET 1               ; stub: always valid

    extract_fields:
        ; Split on commas, parse floats
        RET 0

    compute_gradient:
        RET 0

    depth_normalize:
        RET 0

    accumulate_returns:
        RET 0
