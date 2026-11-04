### Added

- A `@tack.device_class` may define `__tack_missing__(name)`: when a kernel uses a
  method or attribute it lacks, the template engine's error adds the text it
  returns. Expression fields use it to say how to ask for their derivatives.
