extern crate vecmath;

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::iter::from_fn;

pub type Point = self::vecmath::Vector2<i64>;
pub type FPoint = self::vecmath::Vector2<f64>;
pub type Vec3 = self::vecmath::Vector3<i64>;
pub type FVec3 = self::vecmath::Vector3<f64>;
pub type Vec4 = self::vecmath::Vector4<i64>;
pub type FVec4 = self::vecmath::Vector4<f64>;
pub type Mat4 = self::vecmath::Matrix4<i64>;
pub type FMat4 = self::vecmath::Matrix4<f64>;
pub type Mat3 = self::vecmath::Matrix3<i64>;

pub use self::vecmath::vec2_add as point_add;
pub use self::vecmath::vec2_dot as point_dot;
pub use self::vecmath::vec2_neg as point_neg;
pub use self::vecmath::vec2_normalized as point_normalize;
pub use self::vecmath::vec2_scale as point_mul;
pub use self::vecmath::vec2_square_len as point_square_length;
pub use self::vecmath::vec2_sub as point_sub;
pub use self::vecmath::vec3_add as vec_add;
pub use self::vecmath::vec3_cross as vec_cross;
pub use self::vecmath::vec3_dot as vec_dot;
pub use self::vecmath::vec3_neg as vec_neg;

pub const SOUTH: Point = [0, 1];
pub const UP: Point = NORTH;
pub const NORTH_EAST: Point = [1, 1];
pub const UP_RIGHT: Point = NORTH_EAST;
pub const EAST: Point = [1, 0];
pub const RIGHT: Point = EAST;
pub const SOUTH_EAST: Point = [1, -1];
pub const DOWN_RIGHT: Point = SOUTH_EAST;
pub const NORTH: Point = [0, -1];
pub const DOWN: Point = SOUTH;
pub const SOUTH_WEST: Point = [-1, -1];
pub const DOWN_LEFT: Point = SOUTH_WEST;
pub const WEST: Point = [-1, 0];
pub const LEFT: Point = WEST;
pub const NORTH_WEST: Point = [-1, 1];
pub const UP_LEFT: Point = NORTH_WEST;

// Hex directions
// https://www.redblobgames.com/grids/hexagons/
pub const HEX_E: Vec3 = [1, -1, 0];
pub const HEX_W: Vec3 = [-1, 1, 0];
pub const HEX_SE: Vec3 = [0, -1, 1];
pub const HEX_SW: Vec3 = [-1, 0, 1];
pub const HEX_NW: Vec3 = [0, 1, -1];
pub const HEX_NE: Vec3 = [1, 0, -1];

pub const HEX_ALT_SE: Vec3 = [1, -1, 0];
pub const HEX_ALT_NW: Vec3 = [-1, 1, 0];
pub const HEX_ALT_S: Vec3 = [0, -1, 1];
pub const HEX_ALT_SW: Vec3 = [-1, 0, 1];
pub const HEX_ALT_N: Vec3 = [0, 1, -1];
pub const HEX_ALT_NE: Vec3 = [1, 0, -1];

pub const DIRECTIONS: [Point; 4] = [NORTH, EAST, SOUTH, WEST];
pub const DIRECTIONS_INCL_DIAGONALS: [Point; 8] = [
    NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST,
];
pub const HEX_DIRECTIONS: [Vec3; 6] = [HEX_E, HEX_W, HEX_SW, HEX_SE, HEX_NW, HEX_NE];

pub fn neighbors(p: Point) -> impl Iterator<Item = Point> {
    let mut diter = DIRECTIONS.iter();
    from_fn(move || diter.next().map(|d| point_add(p, *d)))
}

pub fn neighbors_incl_diagonals(p: Point) -> impl Iterator<Item = Point> {
    let mut diter = DIRECTIONS_INCL_DIAGONALS.iter();
    from_fn(move || diter.next().map(|d| point_add(p, *d)))
}

pub fn hex_neighbors(p: Vec3) -> impl Iterator<Item = Vec3> {
    let mut diter = HEX_DIRECTIONS.iter();
    from_fn(move || diter.next().map(|d| vec_add(p, *d)))
}

pub fn point_signum(p: Point) -> Point {
    [p[0].signum(), p[1].signum()]
}

pub fn parse_grid<'a, I, J>(lines: I) -> Vec<Vec<char>>
where
    I: IntoIterator<Item = &'a J>,
    J: AsRef<str> + 'a,
{
    lines
        .into_iter()
        .map(|x| AsRef::as_ref(x).chars().collect())
        .collect()
}

pub fn parse_grid_to<'a, I, J, T>(lines: I, f: fn(char) -> T) -> Vec<Vec<T>>
where
    I: IntoIterator<Item = &'a J>,
    J: AsRef<str> + 'a,
{
    lines
        .into_iter()
        .map(|x| AsRef::as_ref(x).chars().map(f).collect())
        .collect()
}

pub fn parse_grid_to_sparse<'a, I, J, T>(lines: I, f: fn(char) -> Option<T>) -> HashMap<Point, T>
where
    I: IntoIterator<Item = &'a J>,
    J: AsRef<str> + 'a,
{
    let mut grid = HashMap::new();
    for (y, line) in lines.into_iter().enumerate() {
        for (x, c) in AsRef::as_ref(line).chars().enumerate() {
            if let Some(t) = f(c) {
                grid.insert([x as i64, y as i64], t);
            }
        }
    }
    grid
}

pub struct GridIteratorHelper {
    extents: (Point, Point),
    curr: Option<Point>,
}

impl Iterator for GridIteratorHelper {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some([x, y]) = self.curr {
            let c = if x < self.extents.1[0] {
                Some([x + 1, y])
            } else if y < self.extents.1[1] {
                Some([self.extents.0[0], y + 1])
            } else {
                None
            };
            let curr = self.curr;
            self.curr = c;
            curr
        } else {
            None
        }
    }
}

pub trait Grid<T>
where
    T: PartialEq + Copy,
{
    fn get_value(&self, pos: Point) -> Option<T>;
    fn set_value(&mut self, pos: Point, value: T);
    fn extents(&self) -> (Point, Point);
    fn points(&self) -> GridIteratorHelper {
        let extents = self.extents();
        GridIteratorHelper {
            extents: extents,
            curr: Some(extents.0),
        }
    }
    fn flip_horizontal(&mut self);
    fn flip_vertical(&mut self);
    fn transpose(&mut self);
    fn rotate_90_cw(&mut self) {
        self.transpose();
        self.flip_horizontal();
    }
    fn rotate_180_cw(&mut self) {
        self.flip_vertical();
        self.flip_horizontal();
    }
    fn rotate_270_cw(&mut self) {
        self.transpose();
        self.flip_vertical();
    }
}

impl<S: ::std::hash::BuildHasher, T> Grid<T> for HashMap<Point, T, S>
where
    T: Clone + Copy + Default + PartialEq,
{
    fn get_value(&self, pos: Point) -> Option<T> {
        self.get(&pos).copied()
    }
    fn set_value(&mut self, pos: Point, value: T) {
        *self.entry(pos).or_insert(value) = value;
    }
    fn extents(&self) -> (Point, Point) {
        let min_x = self.iter().map(|(p, _v)| p[0]).min().unwrap_or(0);
        let min_y = self.iter().map(|(p, _v)| p[1]).min().unwrap_or(0);
        let max_x = self.iter().map(|(p, _v)| p[0]).max().unwrap_or(0);
        let max_y = self.iter().map(|(p, _v)| p[1]).max().unwrap_or(0);
        ([min_x, min_y], [max_x, max_y])
    }
    fn flip_horizontal(&mut self) {
        let ([min_x, _min_y], [max_x, _max_y]) = self.extents();
        let mut new_grid = HashMap::new();
        for ([x, y], v) in self.iter() {
            let new_x = max_x - (x - min_x);
            new_grid.insert([new_x, *y], *v);
        }
        self.clear();
        for (k, v) in new_grid {
            self.insert(k, v);
        }
    }
    fn flip_vertical(&mut self) {
        let ([_min_x, min_y], [_max_x, max_y]) = self.extents();
        let mut new_grid = HashMap::new();
        for ([x, y], v) in self.iter() {
            let new_y = max_y - (y - min_y);
            new_grid.insert([*x, new_y], *v);
        }
        self.clear();
        for (k, v) in new_grid {
            self.insert(k, v);
        }
    }
    fn transpose(&mut self) {
        let mut new_grid = HashMap::new();
        for ([x, y], v) in self.iter() {
            new_grid.insert([*y, *x], *v);
        }
        self.clear();
        for (k, v) in new_grid {
            self.insert(k, v);
        }
    }
}

impl<T> Grid<T> for BTreeMap<Point, T>
where
    T: Clone + Copy + Default + PartialEq,
{
    fn get_value(&self, pos: Point) -> Option<T> {
        self.get(&pos).copied()
    }
    fn set_value(&mut self, pos: Point, value: T) {
        *self.entry(pos).or_insert(value) = value;
    }
    fn extents(&self) -> (Point, Point) {
        let min_x = self.iter().map(|(p, _v)| p[0]).min().unwrap_or(0);
        let min_y = self.iter().map(|(p, _v)| p[1]).min().unwrap_or(0);
        let max_x = self.iter().map(|(p, _v)| p[0]).max().unwrap_or(0);
        let max_y = self.iter().map(|(p, _v)| p[1]).max().unwrap_or(0);
        ([min_x, min_y], [max_x, max_y])
    }
    fn flip_horizontal(&mut self) {
        let ([min_x, _min_y], [max_x, _max_y]) = self.extents();
        let mut new_grid = HashMap::new();
        for ([x, y], v) in self.iter() {
            let new_x = max_x - (x - min_x);
            new_grid.insert([new_x, *y], *v);
        }
        self.clear();
        for (k, v) in new_grid {
            self.insert(k, v);
        }
    }
    fn flip_vertical(&mut self) {
        let ([_min_x, min_y], [_max_x, max_y]) = self.extents();
        let mut new_grid = HashMap::new();
        for ([x, y], v) in self.iter() {
            let new_y = max_y - (y - min_y);
            new_grid.insert([*x, new_y], *v);
        }
        self.clear();
        for (k, v) in new_grid {
            self.insert(k, v);
        }
    }
    fn transpose(&mut self) {
        let mut new_grid = HashMap::new();
        for ([x, y], v) in self.iter() {
            new_grid.insert([*y, *x], *v);
        }
        self.clear();
        for (k, v) in new_grid {
            self.insert(k, v);
        }
    }
}

impl<T> Grid<T> for Vec<Vec<T>>
where
    T: Clone + Copy + Default + PartialEq,
{
    fn get_value(&self, pos: Point) -> Option<T> {
        let [x, y] = pos;
        if let Some(line) = self.get(y as usize) {
            if let Some(p) = line.get(x as usize) {
                return Some(*p);
            }
        }
        None
    }

    fn set_value(&mut self, pos: Point, value: T) {
        let [x, y] = pos;
        if let Some(line) = self.get_mut(y as usize) {
            if let Some(p) = line.get_mut(x as usize) {
                *p = value
            }
        }
    }

    fn extents(&self) -> (Point, Point) {
        if !self.is_empty() && !self[0].is_empty() {
            return (
                [0, 0],
                [(self[0].len() - 1) as i64, (self.len() - 1) as i64],
            );
        }
        ([0, 0], [0, 0])
    }

    fn flip_horizontal(&mut self) {
        let ([minx, miny], [maxx, maxy]) = self.extents();
        let mut new_vec = self.clone();
        for y in miny..=maxy {
            for x in minx..=maxx {
                let v = self[y as usize][x as usize];
                let new_x = maxx - (x - minx);
                new_vec[y as usize][new_x as usize] = v;
            }
        }
        *self = new_vec;
    }

    fn flip_vertical(&mut self) {
        let ([minx, miny], [maxx, maxy]) = self.extents();
        let mut new_vec = self.clone();
        for y in miny..=maxy {
            for x in minx..=maxx {
                let v = self[y as usize][x as usize];
                let new_y = maxy - (y - miny);
                new_vec[new_y as usize][x as usize] = v;
            }
        }
        *self = new_vec;
    }

    fn transpose(&mut self) {
        let ([min_x, min_y], [max_x, max_y]) = self.extents();
        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;
        // Make a vec with the transposed dimensions
        let mut new_vec = Vec::with_capacity(width);
        for _ in min_x..=max_x {
            let mut row = Vec::with_capacity(height);
            row.resize_with(height, Default::default);
            new_vec.push(row);
        }
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let v = self[y as usize][x as usize];
                new_vec[x as usize][y as usize] = v;
            }
        }
        *self = new_vec;
    }
}

pub fn manhattan(n: Point, goal: Point) -> i64 {
    (goal[0] - n[0]).abs() + (goal[1] - n[1]).abs()
}

pub fn manhattan_circumference_plus(p: &Point, manhattan: i64, add: i64) -> Vec<Point> {
    ((p[1] - manhattan - add)..=(p[1] + manhattan + add))
        .into_iter()
        .fold(Vec::<Point>::new(), |mut acc, y| {
            let rest = manhattan - (p[1] - y).abs() + add;
            acc.push([p[0] - rest, y]);
            acc.push([p[0] + rest, y]);
            acc
        })
}

pub fn manhattan_circumference_contains_y(p: &Point, manhattan: i64, y: i64) -> Vec<Point> {
    let rest = manhattan - (p[1] - y).abs();
    ((p[0] - rest)..=(p[0] + rest))
        .into_iter()
        .fold(Vec::<Point>::new(), |mut acc, x| {
            acc.push([x, y]);
            acc
        })
}

pub fn manhattan_vec3(n: Vec3, goal: Vec3) -> i64 {
    (goal[0] - n[0]).abs() + (goal[1] - n[1]).abs() + (goal[2] - n[2]).abs()
}

pub fn manhattan_vec4(n: Vec4, goal: Vec4) -> i64 {
    (goal[0] - n[0]).abs()
        + (goal[1] - n[1]).abs()
        + (goal[2] - n[2]).abs()
        + (goal[3] - n[3]).abs()
}

pub fn manhattan_hex_cube(n: Vec3, goal: Vec3) -> i64 {
    ((goal[0] - n[0]).abs() + (goal[1] - n[1]).abs() + (goal[2] - n[2]).abs()) / 2
}

pub fn astar_grid<T>(
    grid: &dyn Grid<T>,
    is_node: fn(&Point, &T) -> bool,
    get_edge_cost: fn(&Point, &T, &Point, &T) -> Option<i64>,
    start: Point,
    goal: Point,
) -> Option<(i64, Vec<Point>)>
where
    T: PartialEq + Copy,
{
    let mut frontier = BinaryHeap::new();
    let mut came_from = HashMap::new();
    let mut gscore = HashMap::new();
    let mut fscore = HashMap::new();
    gscore.insert(start, 0);
    fscore.insert(start, manhattan(start, goal));
    frontier.push(Reverse((manhattan(start, goal), start)));
    while let Some(Reverse((_est, current))) = frontier.pop() {
        if current == goal {
            let mut path = vec![goal];
            let mut curr = goal;
            while curr != start {
                curr = came_from[&curr];
                path.push(curr)
            }
            return Some((gscore.get_value(goal).unwrap(), path));
        }
        let g = *gscore.entry(current).or_insert(i64::MAX);
        let curr_val = grid.get_value(current).unwrap();
        for nb in neighbors(current) {
            if let Some(value) = grid.get_value(nb) {
                if is_node(&nb, &value) {
                    if let Some(edge_cost) = get_edge_cost(&current, &curr_val, &nb, &value) {
                        let new_g = g + edge_cost;
                        let nb_g = gscore.entry(nb).or_insert(i64::MAX);
                        if new_g < *nb_g {
                            came_from.insert(nb, current);
                            *nb_g = new_g;
                            let new_f = new_g + manhattan(goal, nb);
                            *fscore.entry(nb).or_insert(i64::MAX) = new_f;
                            frontier.push(Reverse((new_f, nb)));
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn dijkstra_grid<T>(
    grid: &dyn Grid<T>,
    is_node: fn(&Point, &T) -> bool,
    get_edge_cost: fn(&Point, &T, &Point, &T) -> Option<i64>,
    start: Point,
    goal: Point,
) -> Option<(i64, Vec<Point>)>
where
    T: PartialEq + Copy,
{
    let mut frontier = BinaryHeap::new();
    let mut visited: HashSet<Point> = HashSet::new();
    let mut came_from = HashMap::new();
    frontier.push(Reverse((0, start)));
    while let Some(Reverse((score, current))) = frontier.pop() {
        if visited.contains(&current) {
            continue;
        }
        if current == goal {
            let mut path = vec![goal];
            let mut curr = goal;
            while curr != start {
                curr = came_from[&curr];
                path.push(curr)
            }
            return Some((score, path.into_iter().rev().collect()));
        }
        let curr_val = grid.get_value(current).unwrap();
        for nb in neighbors(current) {
            if visited.contains(&nb) {
                continue;
            }
            if let Some(value) = grid.get_value(nb) {
                if is_node(&nb, &value) {
                    if let Some(edge_cost) = get_edge_cost(&current, &curr_val, &nb, &value) {
                        let new_score = score + edge_cost;
                        came_from.insert(nb, current);
                        frontier.push(Reverse((new_score, nb)));
                    }
                }
            }
        }
        visited.insert(current);
    }
    None
}

pub fn bfs_grid<T>(
    grid: &dyn Grid<T>,
    is_valid_move: fn(&Point, &T, &Point, &T) -> bool,
    start: Point,
    goal: Point,
) -> Option<Vec<Point>>
where
    T: PartialEq + Copy,
{
    let mut q = VecDeque::<Point>::new();
    let mut visited: HashSet<Point> = HashSet::new();
    let mut came_from = HashMap::new();
    visited.insert(start);
    q.push_back(start);
    while let Some(current) = q.pop_front() {
        if current == goal {
            let mut path = vec![goal];
            let mut curr = came_from[&current];
            while curr != start {
                curr = came_from[&curr];
                path.push(curr);
            }
            return Some(path.into_iter().rev().collect());
        }
        let current_val = grid.get_value(current).unwrap();
        for next in neighbors(current) {
            if visited.contains(&next) {
                continue;
            }
            if let Some(next_val) = grid.get_value(next) {
                if is_valid_move(&current, &current_val, &next, &next_val) {
                    visited.insert(next);
                    came_from.insert(next, current);
                    q.push_back(next);
                }
            }
        }
    }
    None
}

pub trait SliceExt<T> {
    fn partialy_reflects_at(&self, idx: usize) -> bool;
}

impl<T: PartialEq> SliceExt<T> for [T] {
    fn partialy_reflects_at(&self, idx: usize) -> bool {
        let dist = (self.len() - idx).min(idx);

        for i in 0..dist {
            if self[idx - i - 1] != self[idx + i] {
                return false;
            }
        }

        return true;
    }
}

use itertools::Itertools;

pub trait IteratorExt<Item> {
    fn duplicate_positions(self) -> Vec<usize>;
}

impl<Item, Iter: Iterator<Item = Item>> IteratorExt<Item> for Iter
    where
        Item: PartialEq + Clone,
{
    fn duplicate_positions(self) -> Vec<usize> {
        let mut reflections = vec![];

        self.enumerate()
            .tuple_windows()
            .for_each(|((_, prev), (curr_idx, curr))| {
                if prev == curr {
                    reflections.push(curr_idx);
                }
            });

        reflections
    }
}

pub trait UnsignedExt {
    /** https://en.wikipedia.org/wiki/Hamming_distance */
    fn hamming_distance(&self, other: &Self) -> usize;
}

impl UnsignedExt for usize {
    fn hamming_distance(&self, other: &Self) -> usize {
        (self ^ other).count_ones() as usize
    }
}