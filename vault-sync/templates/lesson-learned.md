---
title: "{{ title }}"
type: lesson-learned
created: "{{ created_at }}"
source: "{{ source }}"
tags: [lesson, kai-generated]
conviction: {{ conviction }}
---

# {{ title }}

## Context

{{ context }}

## What happened

{{ what_happened }}

## Lesson

{{ lesson }}

## Linked concepts

{% for concept in concepts %}
- [[{{ concept }}]]
{% endfor %}
