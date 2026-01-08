def get_level(score):
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "moderate"
    else:
        return "high"
    
def fuse_results(dryness, texture, redness, pigmentation, clip_tags):
    attributes = {}
    attributes["dryness"] = get_level(dryness)
    attributes["texture"] = get_level(texture)
    attributes["redness"] = get_level(redness)
    attributes["pigmentation"] = get_level(pigmentation)

    concern_level = "low"
    if pigmentation > 0.65:
        for tag in clip_tags:
            if tag == "localized discoloration":
                concern_level = "elevated"
                break
    
    result = {
        "attributes": attributes,
        "observations" : clip_tags,
        "concern_level" : concern_level,
        "note" : "This analysis is only educational, for a serious medical diagnosis please consult a dermatologist."}
    
    return result