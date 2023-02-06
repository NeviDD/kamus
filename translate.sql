-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 19, 2022 at 03:51 AM
-- Server version: 10.4.25-MariaDB
-- PHP Version: 8.1.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `kamus`
--

-- --------------------------------------------------------

--
-- Table structure for table `translate`
--

CREATE TABLE `translate` (
  `id` int(11) NOT NULL,
  `Indonesia` varchar(225) NOT NULL,
  `Sunda_Lemes` varchar(225) NOT NULL,
  `Sunda_Sedang` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `translate`
--

INSERT INTO `translate` (`id`, `Indonesia`, `Sunda_Lemes`, `Sunda_Sedang`) VALUES
(1, 'makan', 'tuang', 'dahar'),
(2, 'tidur', 'kulem', 'sare'),
(3, 'masuk ke dalam', 'lebet', 'abus'),
(4, 'belum', 'teu acan', 'acan'),
(5, 'adik', 'rai, rayi', 'adi'),
(6, 'adu', 'aben', 'adu'),
(7, 'menggendong', 'emban', 'ais'),
(8, 'untuk', 'haturan, kanggo', 'ajang'),
(10, 'kakak', 'engkang', 'akang'),
(11, 'kakek', 'aki', 'tuang'),
(12, 'mengaku', 'angken', 'aku'),
(13, 'keponakan', 'kapiputra', 'alo'),
(14, 'bagus', 'sae', 'alus'),
(15, 'supaya', 'supados', 'ambeh'),
(16, 'marah', 'bendu', 'ambek'),
(17, 'pamit', 'permios', 'amitan'),
(18, 'bertemu', 'tepang', 'amprok'),
(19, 'anak', 'putra', 'anak'),
(20, 'kerangka', 'cawis', 'anco'),
(21, 'jauh', 'lebih', 'anggang'),
(22, 'bantal', 'bantal', 'anggel'),
(23, 'sudah', 'atos', 'anggeus'),
(24, 'bertamu', 'tatamu', 'anjang'),
(25, 'mahal', 'awis', 'arang'),
(26, 'akan', 'seja', 'arek'),
(27, 'perasaan', 'raos', 'asa'),
(28, 'mula ', 'awit', 'asal'),
(29, 'kuburan', 'makam', 'astana'),
(30, 'atau', 'atanapi', 'atawa'),
(31, 'badan', 'salira', 'awak'),
(32, 'ada', 'aya', 'aya'),
(33, 'jenazah', 'layon', 'babatang'),
(34, 'baca', 'aos', 'baca'),
(35, 'besar', 'ageung', 'badag'),
(36, 'badan', 'salira', 'badan'),
(37, 'mendarat', 'nyacat', 'badarat'),
(38, 'kawan', 'rencang', 'badega'),
(39, 'bagian', 'hancengan', 'bagean'),
(40, 'dahulu', 'kapungkur', 'baheula'),
(41, 'akan', 'seja', 'bakal'),
(42, 'banti', 'baktos', 'bakti'),
(43, 'susah', 'sesah', 'bangga'),
(44, 'bantu', 'bantos', 'bantu'),
(45, 'bapak', 'rama', 'bapa'),
(46, 'saudara', 'wargi', 'baraya'),
(47, 'bersama', 'sareng', 'bareng'),
(48, 'dahulu', 'kapungkur', 'bareto'),
(49, 'waktu', 'waktos', 'basa'),
(50, 'batuk', 'gohgoy', 'batuk'),
(51, 'teman', 'rencang', 'batur'),
(52, 'bawa', 'candak', 'bawa'),
(53, 'kabar', 'unjukan', 'nyanggem'),
(54, 'beda', 'benten', 'beda'),
(55, 'sehat', 'damang', 'cageur'),
(56, 'celana', 'lancingan', 'calanan'),
(57, 'belum', 'teu acan', 'can'),
(58, 'pegal', 'pegel', 'cangkeul'),
(59, 'dagang', 'icalan', 'dagang'),
(60, 'tunggu', 'antos', 'dago'),
(61, 'dapur', 'pawon', 'dapur'),
(62, 'lauk', 'rencang', 'deungen'),
(63, 'paman', 'rama', 'emang'),
(64, 'lama', 'paos', 'endeng'),
(65, 'paman', 'rama', 'emang'),
(66, 'lama', 'paos', 'endeng'),
(67, 'penginapan', 'pajuaran', 'enggon'),
(68, 'iya', 'leres', 'enya'),
(69, 'cepat', 'enggal', 'gancang'),
(70, 'abar', 'lalangse', 'gardeng'),
(71, 'sanggul', 'sanggul', 'gelung'),
(72, 'enak', 'raos', 'genah'),
(73, 'baik', 'sae', 'hade'),
(74, 'hal', 'perkawis', 'hal'),
(75, 'harga', 'pangaos', 'harega'),
(76, 'depan', 'payun', 'hareup'),
(77, 'rumah', 'bumi', 'imah'),
(78, 'pergi', 'angkat', 'indit'),
(79, 'hidung', 'pangambung', 'irung'),
(80, 'tongkat', 'teteken', 'iteuk'),
(81, 'luat', 'anging', 'jaba'),
(82, 'perjaka', 'nonoman', 'jajaka'),
(83, 'untuk', 'haturan', 'jang'),
(84, 'jawab', 'waler', 'jawab'),
(85, 'semuanya', 'sadayana', 'kabehanana'),
(86, 'sangat ', 'kalintang', 'kacida'),
(87, 'saudara', 'wargi', 'kadang'),
(88, 'kedudukan', 'kalungguhan', 'kadudukan'),
(89, 'bukan', 'sanes', 'lain'),
(90, 'kalau', 'upami', 'lamun'),
(91, 'terlewati', 'langkung', 'larung'),
(92, 'lambat', 'lami', 'laun'),
(93, 'lebih', 'langkung ', 'leuwih'),
(94, 'lama', 'lami', 'lila'),
(95, 'baca', 'maos', 'maca'),
(96, 'mahal', 'awis', 'mahal'),
(97, 'maju', 'majeng', 'maju'),
(98, 'maksud', 'maksad', 'maksud'),
(99, 'semoga', 'mugi', 'mangka'),
(100, 'meninggal', 'pupus', 'maot'),
(101, 'sangat', 'kalintang', 'naker'),
(102, 'bertanya', 'naros', 'nanya'),
(103, 'menawar', 'mundut', 'nawar'),
(104, 'melihat', 'ningali', 'nenjo'),
(105, 'mendidik', 'mitutur', 'ngadidik'),
(106, 'penurut', 'tumut', 'ngagugu'),
(107, 'menggendong', 'ngemban', 'ngais'),
(108, 'mengajar', 'ngawulang', 'ngajar'),
(109, 'nama', 'kakasih', 'ngaran'),
(110, 'merokok', 'nyesep', 'ngaroko'),
(111, 'meninggal', 'pupus', 'paeh'),
(112, 'maju', 'pajeng', 'paju'),
(113, 'paman', 'rama', 'paman'),
(114, 'rupa', 'rupi', 'pande'),
(115, 'reda', 'liren', 'raat'),
(116, 'dekat', 'caket', 'raket'),
(117, 'enak', 'raos', 'rasa'),
(118, 'banyak', 'seueur', 'rea'),
(119, 'sering', 'sering', 'remen'),
(120, 'bersama', 'sareng', 'reujeung'),
(121, 'sedia ', 'sayagi', 'sadia'),
(122, 'segala', 'saniskanten', 'sagala'),
(123, 'sebentar', 'sakedap', 'sakeudeung'),
(124, 'sakit', 'kasawat', 'sakit'),
(125, 'takut ', 'salempang', 'salempang'),
(126, 'tukar', 'gentos', 'salin'),
(127, 'perkiraan', 'kinten', 'taksir'),
(128, 'tambah', 'tambih', 'tambah'),
(129, 'terima', 'tampi', 'tampa'),
(130, 'tetangga', 'tanggi', 'tangga'),
(131, 'obat', 'landong', 'ubar'),
(132, 'ucap', 'kedal', 'ucap'),
(133, 'main', 'ameng', 'ulin'),
(134, 'urus', 'lereskeun', 'urus'),
(135, 'masa', 'waktos', 'waktu'),
(136, 'selamat', 'wilujeng', 'waluya'),
(137, 'warga', 'wargi', 'warga'),
(138, 'warna', 'rupi', 'warna');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `translate`
--
ALTER TABLE `translate`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `translate`
--
ALTER TABLE `translate`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=139;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
