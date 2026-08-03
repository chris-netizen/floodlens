# Flooded, and the Shelter Too
### How OpenStreetMap turns a multi-day damage assessment into a five-minute one

On the night of 9–10 September 2024, the Alau Dam collapsed and the Ngadda River poured
into Maiduguri. Roughly 40% of the city went under water. Around 400,000 people were
displaced. And in a cruel twist that defines this disaster: many of the schools that
flooded were also the buildings people fled *to* — at least 18 of the roughly 30
relocation camps set up for the displaced were schools. When a flooded school is also a
shelter, losing it costs twice.

When a flood hits, the first question responders must answer is the hardest to answer
fast: **what got hit?** Which schools are underwater, which clinics can no longer treat
patients, which roads are cut so aid can't get through. Official facility-level damage
figures take days of ground assessment to compile — teams physically inventorying what was
lost, building by building.

But that inventory already exists. **OpenStreetMap** — the free, community-built map of the
world — has already recorded where the schools, clinics, and roads are, before the flood
ever happens. So the slow part is already done. A satellite can show *where the water is* in
minutes; only OSM can tell you that a flooded pixel is a primary school, a health centre, an
evacuation route — and that it matters.

## The tool

**FloodLens** puts those two together. Give it a Sentinel-1 radar image of a flood; it finds
the flooded area, reads the schools, clinics, and roads OSM has already mapped there, and
reports what was exposed — on a map you can download. Radar is simply the fastest way to see
water, through cloud and at night; **OSM is what makes the water mean something.** It works
for *any* flood, anywhere — Maiduguri is the worked example.

## What it found in Maiduguri

Using the official Copernicus Emergency Management flood extent (activation **EMSR753**)
and the infrastructure mapped in OpenStreetMap:

- **7 of 41** mapped schools fell inside the flooded area.
- **16 of 78** mapped clinics and hospitals were exposed.
- **~30,300 buildings** (about 19% of those mapped) stood within the flood.
- **~2,545 road segments** were cut, severing access for aid and evacuation.

Each is a point on the map and a line in a downloadable list — the concrete detail a relief
coordinator can act on the same day.

## Proving it works anywhere

Most floods, in most places, never get an official rapid map. So FloodLens doesn't depend
on one — it builds its own flood extent from free Sentinel-1 radar. For Maiduguri, that
independent map reproduced the official Copernicus delineation with **97.6% containment**:
one person, with free satellite data and a free public map, recreating what a major
emergency service produces. That puts disaster-response capability within anyone's reach.

## The honest gap — and the point

The tool counts **7** flooded schools, but official reports counted **56 schools flooded** —
because OpenStreetMap has only 41 schools mapped across this area. The numbers reflect *what
the open map knows today*, not full ground truth. That gap is the most important finding:
a tool like this is only as complete as the map beneath it, and the map is built by people.
**Mapping your city before the next flood is what lets a tool like this save lives during
it.** OpenStreetMap is a UN-endorsed Digital Public Good — but a public good only delivers
value when people use it. This is one way to use it.

## Limits, honestly — and what's next

Radar sees open floodwater well but struggles inside dense city cores, where water trapped
between buildings can actually look *brighter* to the sensor (a corner-reflector effect).
That's why the urban headline numbers lean on the official extent. The road forward is to
fuse **optical imagery** (which sees urban water directly, when skies are clear) and to add
**learned flood segmentation** trained on labelled SAR floods (Sen1Floods11) — models that
recognise flooding rather than assuming it's dark. FloodLens is the all-weather first
responder; those are its natural next senses.

---

*Data: Copernicus EMS EMSR753 · OpenStreetMap contributors · Sentinel-1 (Copernicus /
Google Earth Engine). Fully reproducible — see README. Built for MapKaton 2026.*
