# Scene Graph Analysis Report — MM-AU

## Purpose
Determine the gold standard for hypergraph construction based on
data-driven evidence from the generated zero-shot scene graphs.

---

## Category 11 — ego-car hitting car
Videos analysed: **3084**

### Most common objects
| Object | Count |
|--------|-------|
| car | 50240 |
| traffic light | 8081 |
| road | 5834 |
| truck | 3340 |
| pedestrian | 2085 |
| bus | 1586 |
| person | 1436 |
| pedestrian person | 1386 |

### Most common semantic relations (zero-shot LLaVA)
| Relation | Count |
|----------|-------|
| colliding_with | 14240 |
| stopped_in_front_of | 8502 |
| yielding_to | 7448 |
| cutting_off | 7110 |
| overtaking | 7060 |
| crossing_path_of | 6952 |
| following | 6915 |
| blocking | 6750 |

### At the collision moment (final frame)
**Objects present:** car(9020), traffic light(1555), road(955), truck(517), pedestrian(412)
**Relations present:** colliding_with(2613), stopped_in_front_of(1511), yielding_to(1350), cutting_off(1279), overtaking(1276)

---

## Hyperedge Design Recommendations
Based on the data above, here are evidence-based proposals for hyperedge types:

| Hyperedge type | Evidence | Static or Learnable? |
|----------------|----------|----------------------|
| **Spatial proximity** | High co-occurrence of near objects at collision | Both |
| **Temporal chain** | Same object tracked across consecutive frames | Static |
| **Collision group** | Objects with 'colliding_with' / 'approaching' at final frame | Learnable |
| **Risk group** | Objects within danger zone (high semantic proximity) | Learnable |
| **Multi-view** | Same event from infra camera (future) | Learnable |

> **Key finding for discussion:** The collision and risk hyperedges are best
> learned rather than hardcoded — the data shows complex multi-object
> interactions that simple geometric rules would miss.