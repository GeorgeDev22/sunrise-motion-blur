use std::time::{Duration, Instant};

/// A frame further apart from the last than this is a new start (a pause, a loading screen):
/// last frame's picture says nothing about this one.
const GAP: Duration = Duration::from_millis(250);

/// How much of each new frame time goes into the smoothed one. Frame times jitter by a
/// millisecond or two even at a steady rate; following them raw would make the streaks breathe.
const SMOOTHING: f32 = 0.15;

/// Measures the time between frames, smoothed.
#[derive(Debug, Default)]
pub struct FrameClock {
    last: Option<Instant>,
    smoothed: Option<f32>,
}

impl FrameClock {
    /// A frame starts at `now`. Returns the smoothed seconds since the last one, or `None` on the
    /// first frame and after a gap, when there is no motion to blur.
    pub fn tick(&mut self, now: Instant) -> Option<f32> {
        let last = self.last.replace(now);
        let elapsed = now.saturating_duration_since(last?);
        if elapsed > GAP {
            self.smoothed = None;
            return None;
        }
        let seconds = elapsed.as_secs_f32().max(1e-4);
        let smoothed = match self.smoothed {
            Some(previous) => previous + (seconds - previous) * SMOOTHING,
            None => seconds,
        };
        self.smoothed = Some(smoothed);
        Some(smoothed)
    }

    /// Forget the last frame, so the next one starts afresh.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn the_first_frame_has_no_time_to_measure() {
        let mut clock = FrameClock::default();
        assert_eq!(clock.tick(Instant::now()), None);
    }

    #[test]
    fn a_steady_rate_reads_as_itself() {
        let mut clock = FrameClock::default();
        let start = Instant::now();
        clock.tick(start);
        let mut seconds = None;
        for frame in 1..=10 {
            seconds = clock.tick(start + ms(16) * frame);
        }
        assert!((seconds.unwrap() - 0.016).abs() < 1e-6);
    }

    #[test]
    fn one_slow_frame_moves_the_time_only_a_little() {
        let mut clock = FrameClock::default();
        let start = Instant::now();
        clock.tick(start);
        clock.tick(start + ms(10));
        let after_hitch = clock.tick(start + ms(60)).unwrap();
        assert!(after_hitch > 0.010 && after_hitch < 0.018, "{after_hitch}");
    }

    #[test]
    fn a_pause_starts_over() {
        let mut clock = FrameClock::default();
        let start = Instant::now();
        clock.tick(start);
        clock.tick(start + ms(10));
        assert_eq!(clock.tick(start + ms(1000)), None);
        assert_eq!(clock.tick(start + ms(1020)), Some(0.020));
    }
}
