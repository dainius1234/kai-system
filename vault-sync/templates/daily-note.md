---
title: "{{ date }}"
type: daily-note
created: "{{ created_at }}"
tags: [daily]
---

# {{ date }}

## Morning intentions

## What Kai noticed today

{% for obs in observations %}
- {{ obs }}
{% endfor %}

## Decisions made

{% for decision in decisions %}
- {{ decision }}
{% endfor %}

## Evening reflection

