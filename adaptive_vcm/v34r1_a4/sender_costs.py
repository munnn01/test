"""Actual search work, separate from logical rows and reused detector outputs."""


def measurement_counts(rows,*,source_calls,controller_calls,repeat_roundtrips,repeat_detector_calls):
    counts=(source_calls,controller_calls,repeat_roundtrips,repeat_detector_calls)
    if any(type(v) is not int or v<0 for v in counts): raise ValueError('nonnegative measured counts required')
    if any(type(r['packet_reused']) is not bool or type(r['decoded_reused']) is not bool for r in rows): raise ValueError('explicit reuse events required')
    roundtrips=sum(not r['packet_reused'] for r in rows)+repeat_roundtrips
    return dict(logical_rows=len(rows),actual_encodes=roundtrips,actual_decodes=roundtrips,
        source_task_model_forwards=source_calls,decoded_task_model_forwards=sum(not r['decoded_reused'] for r in rows)+repeat_detector_calls,
        a2_controller_forwards=controller_calls,repeat_roundtrips=repeat_roundtrips,repeat_detector_forwards=repeat_detector_calls,
        codec_scope='MeasuredCodec performs one encoder and one independent decoder per new packet/roundtrip, including the repeat audit')
