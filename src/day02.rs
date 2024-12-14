use itertools::Itertools;

#[aoc_generator(day2)]
pub fn generator(input: &str) -> Vec<Vec<isize>> {
    input
        .lines()
        .map(|line| line.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect()
}

#[aoc(day2, part1)]
pub fn part1(input: &Vec<Vec<isize>>) -> isize {
    input.iter().map(|row| {
        let diff: Vec<isize> = row
            .iter()
            .tuple_windows()
            .map(|(a, b)| b - a).collect();
        match is_valid(&diff) {
            true => 1,
            false => 0
        }
    }).sum()
}

#[aoc(day2, part2)]
pub fn part2(input: &Vec<Vec<isize>>) -> isize {
    input.iter().map(|row| {
        match (0..row.len()).map(|i| {
            let mut current = row.clone();
            current.remove(i);
            let diff: Vec<isize> = current
                .iter()
                .tuple_windows()
                .map(|(a, b)| b - a).collect();
            is_valid(&diff)
        }).any(|x| x) {
            true => 1,
            false => 0
        }
    }).sum()
}

pub fn is_valid(input: &Vec<isize>) -> bool {
    let monotonic = input.iter().map(|&x| {  match x {
        x if x > 0 => 1,
        x if x < 0 => -1,
        _ => 0
    } }).unique().count() == 1;
    let delta_range = input.iter().all(|&x| x.abs() >= 1 && x.abs() <= 3);
    monotonic && delta_range
}
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9";

    #[test]
    pub fn test1() {
        assert_eq!(part1(&generator(&SAMPLE)), 2)
    }

    #[test]
    pub fn test2() {
        assert_eq!(part2(&generator(&SAMPLE)), 4)
    }
}
